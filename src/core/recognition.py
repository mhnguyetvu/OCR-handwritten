from core.models import ModelSuite

def recognize_fields(detection_data):
    rec = ModelSuite.get_recognizer()
    texts = rec.recognize_from_detection(detection_data)
    raw_text = "\n".join(texts)
    fields = rec.extract_fields(raw_text)
    fields["seal_present"] = rec.detect_seal_from_bboxes(detection_data)
    return texts, fields
