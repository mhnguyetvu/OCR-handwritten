# utils/text_utils.py
import re

def clean_line(s: str) -> str:
    """Normalize OCR line text: remove odd chars, collapse whitespace."""
    if not s:
        return ""
    s = str(s)
    s = s.replace("\t", " ").replace("\n", " ").replace("•", " ").replace("·", " ")
    # keep printable + many unicode letters
    s = "".join(ch for ch in s if (31 < ord(ch) < 127) or ord(ch) > 160)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s
