import os
import logging
import numpy as np
import fitz  # PyMuPDF

from core.ocr_engine import ocr_image, _detect_language

logger = logging.getLogger(__name__)

_DPI_SCALE = 300 / 72
_NATIVE_TEXT_THRESHOLD = 50


def extract_from_pdf(file_path: str) -> str:
    """
    Hybrid PDF extraction:
      1. Native text via PyMuPDF — instant, zero GPU usage for digital PDFs.
      2. PaddleOCR fallback — only for pages with no embedded text (scanned pages).

    Language is detected from the first OCR page and reused for all subsequent OCR pages.
    """
    abs_path = os.path.abspath(file_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"File not found: {abs_path}")

    doc = fitz.open(abs_path)
    text_parts: list[str] = []
    detected_lang: str | None = None

    try:
        for page_num, page in enumerate(doc):
            native_text = page.get_text().strip()
            if len(native_text) >= _NATIVE_TEXT_THRESHOLD:
                text_parts.append(native_text)
                logger.debug("PDF page %d/%d: native text (%d chars).", page_num + 1, len(doc), len(native_text))
                continue

            logger.debug("PDF page %d/%d: scanned — running OCR.", page_num + 1, len(doc))
            mat = fitz.Matrix(_DPI_SCALE, _DPI_SCALE)
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

            page_text = ocr_image(img_array, hint_lang=detected_lang)

            if detected_lang is None and page_text.strip():
                detected_lang = _detect_language(page_text)

            if page_text.strip():
                text_parts.append(page_text)
    finally:
        doc.close()

    return "\n".join(text_parts)
