# extractors/field_extractor.py
import re

def extract_fields_from_lines(lines):
    """
    Given OCR lines, return a dict with keys:
    decision_number, decision_date, appointee_name, position, term, company
    """
    joined = "\n".join(lines)
    out = {}

    # decision_number
    m = re.search(r"\b\d{1,4}\s*\/\s*[A-ZĐÂĂƠƯ0-9\-]{1,40}\b", joined, flags=re.I)
    if m:
        out["decision_number"] = m.group(0).replace(" ", "")

    # decision_date
    m = re.search(r"\b([0-3]?\d[\/\-\.\s][0-1]?\d[\/\-\.\s](19|20)\d{2})\b", joined)
    if m:
        out["decision_date"] = m.group(1).replace(" ", "")

    # term
    m = re.search(r"\b(19|20)\d{2}\s*[-–]\s*(19|20)\d{2}\b", joined)
    if m:
        out["term"] = m.group(0).replace(" ", "")

    # company
    for line in lines[:10]:
        if re.search(r"\b(CÔNG TY|CONG TY|CTY|CÔNG TY CP|CÔNG TY CỔ PHẦN)\b", line, flags=re.I):
            out["company"] = re.sub(r"[^A-Za-z0-9\u00C0-\u1EF9\s\.\,\/\-\&\(\)]", "", line).strip()
            break
    if "company" not in out:
        for line in lines:
            if re.search(r"\b(CÔNG TY|CONG TY|CTY)\b", line, flags=re.I):
                out["company"] = re.sub(r"[^A-Za-z0-9\u00C0-\u1EF9\s\.\,\/\-\&\(\)]", "", line).strip()
                break

    # position
    positions = ["Tổng Giám đốc", "Giám đốc", "Phó Giám đốc", "Kế toán trưởng",
                 "Chủ tịch", "Thư ký", "Thành viên Hội đồng"]
    pos_regex = "|".join([re.escape(p) for p in positions])
    m = re.search(r"\b(" + pos_regex + r")\b", joined, flags=re.I)
    if m:
        out["position"] = m.group(0)

    # appointee_name: heuristics
    name = None
    # heuristic A: Ông/Bà followed by capitalized tokens
    for line in lines:
        m = re.search(r"\b(Ông|Bà)\s+([A-Z\u00C0-\u1EF9][^\d\,\(\)\n]{2,60})", line, flags=re.I)
        if m:
            cand = m.group(2).strip()
            cand = re.split(r"(sinh|năm sinh|,|\()", cand, maxsplit=1, flags=re.I)[0].strip()
            cand = re.sub(r"[^A-Za-z\u00C0-\u1EF9\s\-']", " ", cand).strip()
            if len(cand.split()) >= 2:
                name = cand
                break
    # heuristic B: lines with 'Bổ nhiệm' find capitalized sequences
    if not name:
        for line in lines:
            if re.search(r"b[ổo] nhi[ệe]m|b nhim|bo nhiem", line, flags=re.I):
                tokens = re.findall(r"[A-Z\u00C0-\u1EF9][a-z\u00C0-\u1EF9]{1,20}(?:\s+[A-Z\u00C0-\u1EF9][a-z\u00C0-\u1EF9]{1,20}){0,3}", line)
                if tokens:
                    name = max(tokens, key=lambda s: len(s.split()))
                    break
    # heuristic C: Ho ten / Ten
    if not name:
        for line in lines:
            m = re.search(r"\b(Họ tên|Họ và tên|Tên|Ho ten)\b[:\s\-]*([A-Z\u00C0-\u1EF9][^\n]{2,60})", line, flags=re.I)
            if m:
                cand = m.group(2)
                cand = re.split(r"(,|\(|\n|sinh|năm sinh)", cand, maxsplit=1, flags=re.I)[0].strip()
                cand = re.sub(r"[^A-Za-z\u00C0-\u1EF9\s\-']", " ", cand).strip()
                if len(cand.split()) >= 2:
                    name = cand
                    break
    if name:
        out["appointee_name"] = name

    return out
