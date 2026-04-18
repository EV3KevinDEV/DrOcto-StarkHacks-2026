import asyncio
import base64
import logging
import os
from typing import Callable, Optional

import sounddevice as sd
from elevenlabs import (
    AudioFormat,
    CommitStrategy,
    ElevenLabs,
    RealtimeAudioOptions,
    RealtimeEvents,
)

logger = logging.getLogger(__name__)


class ElevenLabsRealtimeTranscriber:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: str = "scribe_v2_realtime",
        sample_rate: int = 16000,
        channels: int = 1,
        block_ms: int = 100,
        commit_strategy: CommitStrategy = CommitStrategy.MANUAL,
        include_timestamps: bool = True,
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
        self.commit_strategy = commit_strategy
        self.include_timestamps = include_timestamps

        self.on_partial = on_partial
        self.on_commit = on_commit
        self.on_error = on_error

        self.client = ElevenLabs(api_key=self.api_key)
        self.connection = None

        self.audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.stop_event = asyncio.Event()
        self.done_event = asyncio.Event()

        self._audio_stream = None
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        self.connection = await self.client.speech_to_text.realtime.connect(
            RealtimeAudioOptions(
                model_id=self.model_id,
                audio_format=AudioFormat.PCM_16000,
                sample_rate=self.sample_rate,
                commit_strategy=self.commit_strategy,
                include_timestamps=self.include_timestamps,
                vad_silence_threshold_secs=1.5,
                vad_threshold=0.4,
                min_speech_duration_ms=100,
                min_silence_duration_ms=100,
            )
        )

        self._register_handlers()
        if self.commit_strategy == CommitStrategy.MANUAL:
            self._tasks.append(asyncio.create_task(self._commit_loop()))

    async def run_forever(self) -> None:
        await self.start()
        try:
            await self.done_event.wait()
        except KeyboardInterrupt:
            logger.info("Stopping...")
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
                pass

        if self.connection is not None:
            try:
                await self.connection.close()
            except Exception:
                pass

        self.done_event.set()

    async def commit(self) -> None:
        if self.connection is not None:
            await self.connection.commit()

    def _register_handlers(self) -> None:
        self.connection.on(RealtimeEvents.SESSION_STARTED, self._on_session_started)
        self.connection.on(RealtimeEvents.PARTIAL_TRANSCRIPT, self._on_partial_transcript)
        self.connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT, self._on_committed_transcript)
        self.connection.on(
            RealtimeEvents.COMMITTED_TRANSCRIPT_WITH_TIMESTAMPS,
            self._on_committed_transcript_with_timestamps,
        )
        self.connection.on(RealtimeEvents.ERROR, self._on_error_event)
        self.connection.on(RealtimeEvents.CLOSE, self._on_close)

    def _on_session_started(self, data) -> None:
        logger.debug("Session started: %s", data)
        self._start_audio_stream()
        self._tasks.append(asyncio.create_task(self._stream_mic_audio()))

    def _on_partial_transcript(self, data) -> None:
        transcript = data.get("text", "").strip()
        if transcript:
            logger.debug("Partial: %s", transcript)
            if self.on_partial:
                self.on_partial(transcript)

    def _on_committed_transcript(self, data) -> None:
        transcript = data.get("text", "").strip()
        if transcript:
            logger.debug("Committed transcript: %s", transcript)
            if self.on_commit:
                self.on_commit(transcript)

    def _on_committed_transcript_with_timestamps(self, data) -> None:
        words = data.get("words", [])
        if words:
            logger.debug("Timestamps: %s", words)
            logger.debug("%s", "-" * 50)

    def _on_error_event(self, error) -> None:
        logger.debug("Realtime error: %s", error)
        if self.on_error:
            self.on_error(error)
        self.done_event.set()

    def _on_close(self) -> None:
        logger.debug("Connection closed")
        self.done_event.set()

    def _start_audio_stream(self) -> None:
        def audio_callback(indata, frames, time_info, status):
            if status:
                logger.debug("[audio status] %s", status)

            if not indata.any():
                logger.debug("[audio] empty chunk")

            try:
                self.audio_queue.put_nowait(indata.tobytes())
            except asyncio.QueueFull:
                pass

        self._audio_stream = sd.InputStream(
            device="Brio 105",
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=self.block_size,
            callback=audio_callback,
        )
        self._audio_stream.start()
        logger.debug("[audio] mic stream started")

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
        except Exception as e:
            logger.exception("Error streaming mic audio: %s", e)
            if self.on_error:
                self.on_error(e)
            self.done_event.set()

    async def _commit_loop(self) -> None:
        try:
            while not self.stop_event.is_set():
                await asyncio.to_thread(
                    input,
                    "\nPress Enter to commit current utterance...\n",
                )
                await self.commit()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception("Error in commit loop: %s", e)
            if self.on_error:
                self.on_error(e)
            self.done_event.set()