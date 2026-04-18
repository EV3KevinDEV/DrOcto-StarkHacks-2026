"""CommandSource backed by ElevenLabs realtime STT.

This bridges the async transcriber (mic -> ElevenLabs -> committed transcripts)
into the synchronous CommandSource.next_command() seam the SessionWorker uses.

Lifecycle is managed explicitly: ``start()`` launches the transcriber on a
background asyncio loop, ``stop()`` tears it down. DocOckRuntime ties these
to the voice-mode toggle so the mic is only live when the user asks for it.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

from doc_ock.commands import TaskCommand

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioConfig:
    enabled: bool = False
    device: Optional[str | int] = None
    commit_strategy: str = "vad"
    model_id: str = "scribe_v2_realtime"
    sample_rate: int = 16000


class ElevenLabsCommandSource:
    def __init__(
        self,
        config: AudioConfig,
        on_partial: Optional[Callable[[str], None]] = None,
        on_commit: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[object], None]] = None,
    ) -> None:
        self._config = config

        self._external_on_partial = on_partial
        self._external_on_commit = on_commit
        self._external_on_error = on_error

        self._queue: "queue.Queue[TaskCommand]" = queue.Queue()
        self._state_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._transcriber: Any = None
        self._ready_event = threading.Event()
        self._started = False

        self._latest_partial: str = ""
        self._latest_committed: Optional[str] = None
        self._latest_error: Optional[str] = None

    @property
    def latest_partial(self) -> str:
        with self._state_lock:
            return self._latest_partial

    @property
    def latest_committed(self) -> Optional[str]:
        with self._state_lock:
            return self._latest_committed

    @property
    def latest_error(self) -> Optional[str]:
        with self._state_lock:
            return self._latest_error

    @property
    def is_running(self) -> bool:
        with self._state_lock:
            return self._started

    def next_command(self, timeout_s: Optional[float] = None) -> Optional[TaskCommand]:
        try:
            if timeout_s is None or timeout_s <= 0:
                return self._queue.get_nowait()
            return self._queue.get(timeout=timeout_s)
        except queue.Empty:
            return None

    def start(self) -> None:
        with self._state_lock:
            if self._started:
                return
            self._started = True
            self._latest_error = None

        self._ready_event.clear()
        self._thread = threading.Thread(
            target=self._thread_main, name="doc-ock-audio", daemon=True
        )
        self._thread.start()

        if not self._ready_event.wait(timeout=10.0):
            logger.warning("Transcriber did not signal ready within 10s")

    def stop(self) -> None:
        with self._state_lock:
            if not self._started:
                return
            self._started = False

        loop = self._loop
        transcriber = self._transcriber
        if loop is not None and loop.is_running() and transcriber is not None:
            try:
                fut = asyncio.run_coroutine_threadsafe(transcriber.stop(), loop)
                fut.result(timeout=3.0)
            except Exception:
                logger.exception("Error requesting transcriber stop")

        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)

        self._thread = None
        self._loop = None
        self._transcriber = None
        self._ready_event.clear()

    def request_manual_commit(self) -> None:
        transcriber = self._transcriber
        if transcriber is not None:
            try:
                transcriber.request_manual_commit()
            except Exception:
                logger.exception("Manual commit failed")

    def _on_partial(self, text: str) -> None:
        with self._state_lock:
            self._latest_partial = text
        if self._external_on_partial is not None:
            try:
                self._external_on_partial(text)
            except Exception:
                logger.exception("external on_partial failed")

    def _on_commit(self, text: str) -> None:
        stripped = (text or "").strip()
        if not stripped:
            return
        with self._state_lock:
            self._latest_committed = stripped
            self._latest_partial = ""
        self._queue.put(TaskCommand(task_text=stripped))
        logger.info("Committed transcript -> task: %s", stripped)
        if self._external_on_commit is not None:
            try:
                self._external_on_commit(stripped)
            except Exception:
                logger.exception("external on_commit failed")

    def _on_error(self, err: Any) -> None:
        with self._state_lock:
            self._latest_error = str(err)
        if self._external_on_error is not None:
            try:
                self._external_on_error(err)
            except Exception:
                logger.exception("external on_error failed")

    def _thread_main(self) -> None:
        try:
            from doc_ock.audio.elevenlabs_transcriber import ElevenLabsRealtimeTranscriber
        except Exception as exc:
            logger.error("Audio dependencies unavailable: %s", exc)
            self._on_error(exc)
            self._ready_event.set()
            with self._state_lock:
                self._started = False
            return

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop

        try:
            transcriber = ElevenLabsRealtimeTranscriber(
                model_id=self._config.model_id,
                sample_rate=self._config.sample_rate,
                commit_strategy=self._config.commit_strategy,
                input_device=self._config.device,
                on_partial=self._on_partial,
                on_commit=self._on_commit,
                on_error=self._on_error,
            )
            self._transcriber = transcriber
        except Exception as exc:
            logger.error("Failed to construct transcriber: %s", exc)
            self._on_error(exc)
            self._ready_event.set()
            loop.close()
            with self._state_lock:
                self._started = False
            return

        self._ready_event.set()
        try:
            loop.run_until_complete(transcriber.run_forever())
        except Exception as exc:
            logger.exception("Transcriber loop crashed")
            self._on_error(exc)
        finally:
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
            except Exception:
                pass
            loop.close()
            with self._state_lock:
                self._started = False


__all__ = ["AudioConfig", "ElevenLabsCommandSource"]
