"""ElevenLabs realtime STT transcriber, ported from the speech-to-text branch.

Changes vs. upstream snippet:
- `input_device` is configurable (defaults to system default; branch hardcoded "Brio 105").
- Removes the blocking stdin commit loop; manual commit is now an explicit
  `await commit()` call or via `request_manual_commit()` from another thread.
- Uses a thread-safe bridge for mic-callback -> asyncio queue.
- All secret/env reads happen at construction, not import time.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class ElevenLabsRealtimeTranscriber:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: str = "scribe_v2_realtime",
        sample_rate: int = 16000,
        channels: int = 1,
        block_ms: int = 100,
        commit_strategy: str = "vad",
        include_timestamps: bool = True,
        input_device: Optional[str | int] = None,
        on_partial: Optional[Callable[[str], None]] = None,
        on_commit: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[object], None]] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY is not set")

        self.model_id = model_id
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_ms = block_ms
        self.block_size = self.sample_rate * self.block_ms // 1000
        self.commit_strategy_name = commit_strategy.lower()
        self.include_timestamps = include_timestamps
        self.input_device = input_device

        self.on_partial = on_partial
        self.on_commit = on_commit
        self.on_error = on_error

        self.connection: Any = None
        self._client: Any = None

        self.audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.stop_event = asyncio.Event()
        self.done_event = asyncio.Event()

        self._audio_stream: Any = None
        self._tasks: list[asyncio.Task] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def start(self) -> None:
        from elevenlabs import (
            AudioFormat,
            CommitStrategy,
            ElevenLabs,
            RealtimeAudioOptions,
        )

        strategy = {
            "manual": CommitStrategy.MANUAL,
            "vad": getattr(CommitStrategy, "VAD", CommitStrategy.MANUAL),
            "auto": getattr(CommitStrategy, "AUTO", CommitStrategy.MANUAL),
        }.get(self.commit_strategy_name, CommitStrategy.MANUAL)

        self._client = ElevenLabs(api_key=self.api_key)
        self._loop = asyncio.get_running_loop()

        self.connection = await self._client.speech_to_text.realtime.connect(
            RealtimeAudioOptions(
                model_id=self.model_id,
                audio_format=AudioFormat.PCM_16000,
                sample_rate=self.sample_rate,
                commit_strategy=strategy,
                include_timestamps=self.include_timestamps,
                vad_silence_threshold_secs=1.5,
                vad_threshold=0.4,
                min_speech_duration_ms=100,
                min_silence_duration_ms=100,
            )
        )

        self._register_handlers()

    async def run_forever(self) -> None:
        await self.start()
        try:
            await self.done_event.wait()
        except KeyboardInterrupt:
            logger.info("Stopping transcriber")
        finally:
            await self.stop()

    async def stop(self) -> None:
        if self.stop_event.is_set():
            return

        self.stop_event.set()

        for task in self._tasks:
            task.cancel()

        if self._audio_stream is not None:
            try:
                self._audio_stream.stop()
                self._audio_stream.close()
            except Exception:
                logger.exception("Error closing audio stream")

        if self.connection is not None:
            try:
                await self.connection.close()
            except Exception:
                logger.exception("Error closing ElevenLabs connection")

        self.done_event.set()

    async def commit(self) -> None:
        if self.connection is not None:
            await self.connection.commit()

    def request_manual_commit(self) -> None:
        """Thread-safe manual commit request."""
        if self._loop is not None and self.connection is not None:
            asyncio.run_coroutine_threadsafe(self.commit(), self._loop)

    def _register_handlers(self) -> None:
        from elevenlabs import RealtimeEvents

        self.connection.on(RealtimeEvents.SESSION_STARTED, self._on_session_started)
        self.connection.on(RealtimeEvents.PARTIAL_TRANSCRIPT, self._on_partial_transcript)
        self.connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT, self._on_committed_transcript)
        self.connection.on(
            RealtimeEvents.COMMITTED_TRANSCRIPT_WITH_TIMESTAMPS,
            self._on_committed_transcript_with_timestamps,
        )
        self.connection.on(RealtimeEvents.ERROR, self._on_error_event)
        self.connection.on(RealtimeEvents.CLOSE, self._on_close)

    def _on_session_started(self, data: Any) -> None:
        logger.debug("Session started: %s", data)
        self._start_audio_stream()
        self._tasks.append(asyncio.create_task(self._stream_mic_audio()))

    def _on_partial_transcript(self, data: Any) -> None:
        transcript = (data.get("text") or "").strip()
        if transcript and self.on_partial:
            try:
                self.on_partial(transcript)
            except Exception:
                logger.exception("on_partial handler failed")

    def _on_committed_transcript(self, data: Any) -> None:
        transcript = (data.get("text") or "").strip()
        if transcript and self.on_commit:
            try:
                self.on_commit(transcript)
            except Exception:
                logger.exception("on_commit handler failed")

    def _on_committed_transcript_with_timestamps(self, data: Any) -> None:
        words = data.get("words", []) if isinstance(data, dict) else []
        logger.debug("Timestamps: %s", words)

    def _on_error_event(self, error: Any) -> None:
        logger.warning("Realtime transcriber error: %s", error)
        if self.on_error:
            try:
                self.on_error(error)
            except Exception:
                logger.exception("on_error handler failed")
        self.done_event.set()

    def _on_close(self) -> None:
        logger.debug("Realtime connection closed")
        self.done_event.set()

    def _start_audio_stream(self) -> None:
        import sounddevice as sd

        loop = self._loop
        assert loop is not None

        def audio_callback(indata, frames, time_info, status):
            if status:
                logger.debug("[audio status] %s", status)
            chunk = bytes(indata)
            try:
                loop.call_soon_threadsafe(self.audio_queue.put_nowait, chunk)
            except RuntimeError:
                pass

        stream_kwargs = dict(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=self.block_size,
            callback=audio_callback,
        )
        if self.input_device is not None:
            stream_kwargs["device"] = self.input_device

        self._audio_stream = sd.InputStream(**stream_kwargs)
        self._audio_stream.start()
        logger.info(
            "Mic stream started (device=%s sample_rate=%d block_ms=%d)",
            self.input_device if self.input_device is not None else "<default>",
            self.sample_rate,
            self.block_ms,
        )

    async def _stream_mic_audio(self) -> None:
        try:
            while not self.stop_event.is_set():
                chunk = await self.audio_queue.get()
                chunk_base64 = base64.b64encode(chunk).decode("utf-8")
                await self.connection.send(
                    {
                        "audio_base_64": chunk_base64,
                        "sample_rate": self.sample_rate,
                    }
                )
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.exception("Error streaming mic audio")
            if self.on_error:
                self.on_error(exc)
            self.done_event.set()
