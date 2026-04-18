import logging


logger = logging.getLogger(__name__)

class VLAModel:
    def handle_text(self, text: str) -> None:
        logger.info("[VLA INPUT] %s", text)