import asyncio
import logging

from dotenv import load_dotenv

from vla.vla import VLAModel
from speech_to_text.speech_to_text import ElevenLabsRealtimeTranscriber


load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)

async def main() -> None:
    vla = VLAModel()

    transcriber = ElevenLabsRealtimeTranscriber(
        on_partial=lambda text: None,
        on_commit=vla.handle_text,
        on_error=lambda err: logger.error("[TRANSCRIBER ERROR] %s", err),
    )

    await transcriber.run_forever()


if __name__ == "__main__":
    asyncio.run(main())