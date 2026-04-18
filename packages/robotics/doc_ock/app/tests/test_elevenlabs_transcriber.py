from __future__ import annotations

import numpy as np

from doc_ock.audio.elevenlabs_transcriber import ElevenLabsRealtimeTranscriber


def test_noise_gate_zeroes_quiet_chunks() -> None:
    transcriber = ElevenLabsRealtimeTranscriber(api_key="test-key")
    quiet_chunk = np.zeros(1600, dtype=np.int16).tobytes()

    gated = transcriber._gate_audio_chunk(quiet_chunk)

    assert gated == bytes(len(quiet_chunk))


def test_noise_gate_keeps_voiced_chunk_and_short_hangover() -> None:
    transcriber = ElevenLabsRealtimeTranscriber(api_key="test-key")
    voice_chunk = np.full(1600, 2000, dtype=np.int16).tobytes()
    quiet_chunk = np.zeros(1600, dtype=np.int16).tobytes()

    first = transcriber._gate_audio_chunk(voice_chunk)
    second = transcriber._gate_audio_chunk(quiet_chunk)

    assert first == voice_chunk
    assert second == quiet_chunk
