import re


def clean_text(text: str) -> str:
    """
    Clean extracted text while preserving document structure.
    - Removes excessive whitespace but keeps paragraph breaks
    - Strips page numbers, headers/footers
    - Normalizes unicode characters
    """
    if not text:
        return ""
    
    text = text.replace('\xa0', ' ')
    text = text.replace('\u200b', '')
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    text = re.sub(r'\n\s*\d{1,3}\s*\n', '\n', text)
    
    text = re.sub(r'[-_=]{10,}', '', text)
    
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    text = re.sub(r'[^\S\n]+', ' ', text)
    
    lines = text.split('\n')
    cleaned_lines = [line.strip() for line in lines]
    text = '\n'.join(cleaned_lines)
    
    return text.strip()
