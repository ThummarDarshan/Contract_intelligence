import logging
import numpy as np
from PIL import Image

from core.ocr_engine import ocr_image

logger = logging.getLogger(__name__)


def extract_from_image(file_path: str) -> str:
    """
    Extracts text from an image file using PaddleOCR.
    angle_cls=True is kept here because camera/phone images may be rotated.
    """
    img = Image.open(file_path).convert("RGB")
    img_array = np.array(img)
    text = ocr_image(img_array, hint_lang=None, angle_cls=True)
    logger.debug("Image '%s': extracted %d chars.", file_path, len(text))
    return text