import os
# MUST BE SET BEFORE IMPORTING PADDLE/PADDLEOCR
os.environ["FLAGS_allocator_strategy"] = "auto_growth"
os.environ["FLAGS_fraction_of_gpu_memory_to_use"] = "0.1"  # Only pre-allocate 10% initially

import logging
from functools import lru_cache
from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)

def _detect_gpu() -> bool:
    """Returns True if paddlepaddle-gpu is installed and a CUDA GPU is found."""
    try:
        import paddle
        if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            # Free any pre-allocated memory on startup
            paddle.device.cuda.empty_cache()
            logger.info("PaddleOCR: %d GPU(s) detected — CUDA enabled.", paddle.device.cuda.device_count())
            return True
        logger.info("PaddleOCR: No CUDA GPU found — using CPU.")
        return False
    except Exception as exc:
        logger.warning("PaddleOCR: GPU detection failed (%s) — using CPU.", exc)
        return False


USE_GPU: bool = _detect_gpu()

# Maps langdetect language codes → PaddleOCR lang codes.
# Unlisted languages fall back to 'en'.
_LANG_MAP: dict[str, str] = {
    "zh-cn": "ch",
    "zh-tw": "chinese_cht",
    "fr":    "fr",
    "de":    "german",
    "ja":    "japan",
    "ko":    "korean",
    "ar":    "ar",
    "hi":    "hi",
    "pt":    "pt",
    "ru":    "ru",
    "es":    "es",
    "it":    "it",
    "uk":    "uk",
    "vi":    "vi",
    "tr":    "tr",
    "pl":    "pl",
}


@lru_cache(maxsize=16)
def _get_engine(lang: str = "en") -> PaddleOCR:
    """Returns a cached PaddleOCR engine for the given language. Models are downloaded once and cached."""
    logger.info("PaddleOCR: loading engine lang='%s' gpu=%s.", lang, USE_GPU)
    return PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=USE_GPU, show_log=False)


# Pre-warm English engine so the first real request isn't delayed by model loading.
_get_engine("en")


def _detect_language(text: str) -> str:
    """Detects the language of text and returns the matching PaddleOCR lang code. Returns 'en' on failure."""
    if not text or len(text.strip()) < 20:
        return "en"
    try:
        from langdetect import detect
        raw = detect(text)
        lang = _LANG_MAP.get(raw, "en")
        if lang != "en":
            logger.info("Language detected: '%s' → PaddleOCR lang='%s'.", raw, lang)
        return lang
    except Exception as exc:
        logger.debug("Language detection failed (%s) — defaulting to 'en'.", exc)
        return "en"


def _join_result(result) -> str:
    """Joins raw PaddleOCR result lines into a single string."""
    return "\n".join(line[1][0] for page in (result or []) if page for line in page)


def ocr_image(img_array, hint_lang: str | None = None) -> str:
    """
    Runs OCR on a numpy image array with automatic language detection.
    """
    try:
        initial_lang = hint_lang or "en"
        first_text = _join_result(_get_engine(initial_lang).ocr(img_array, cls=True))

        detected_lang = _detect_language(first_text)
        if detected_lang != initial_lang:
            logger.info("Re-running OCR with lang='%s'.", detected_lang)
            second_text = _join_result(_get_engine(detected_lang).ocr(img_array, cls=True))
            return second_text if len(second_text) >= len(first_text) else first_text

        return first_text
    finally:
        # ALWAYS release GPU memory back to Windows after processing a chunk
        if USE_GPU:
            import paddle
            paddle.device.cuda.empty_cache()
