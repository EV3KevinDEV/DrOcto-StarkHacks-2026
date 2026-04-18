from __future__ import annotations

from doc_ock.voice_mode import VoiceModeController


def test_voice_mode_get_set_toggle() -> None:
    controller = VoiceModeController()

    assert controller.get_enabled() is False
    assert controller.set_enabled(True) is True
    assert controller.get_enabled() is True
    assert controller.toggle() is False
    assert controller.get_enabled() is False
