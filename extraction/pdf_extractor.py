import os
import logging
import numpy as np
import fitz  # PyMuPDF

from core.ocr_engine import ocr_image, _detect_language

logger = logging.getLogger(__name__)

_DPI_SCALE = 300 / 72  # PyMuPDF uses 72 pt/inch; 300 DPI gives good OCR accuracy


def extract_from_pdf(file_path: str) -> str:
    """
    Extracts text from a PDF using PyMuPDF (page rendering) + PaddleOCR.
    Language is detected on page 1 and reused for all remaining pages.
    """
    abs_path = os.path.abspath(file_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"File not found: {abs_path}")

    doc = fitz.open(abs_path)
    text_parts: list[str] = []
    detected_lang: str | None = None

    try:
        for page_num, page in enumerate(doc):
            mat = fitz.Matrix(_DPI_SCALE, _DPI_SCALE)
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

            if page_num == 0:
                page_text = ocr_image(img_array, hint_lang=None)
                detected_lang = _detect_language(page_text)
            else:
                page_text = ocr_image(img_array, hint_lang=detected_lang)

            if page_text.strip():
                text_parts.append(page_text)

            logger.debug("PDF page %d/%d done.", page_num + 1, len(doc))
    finally:
        doc.close()

    return "\n".join(text_parts)
