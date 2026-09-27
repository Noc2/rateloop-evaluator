"""Frozen portable task contracts. Any structural change requires a new version."""
import unicodedata

from .protocol import Template, commitment

CUSTOM_TEXT_CAPABILITY = {"schemaVersion": "rateloop.evaluator.custom-binary-text.v1"}


def overall_approval(language: str = "en") -> Template:
    if language not in ("en","de"):
        raise ValueError("Overall approval supports English and German")
    prompt,positive,negative={
        "en":("Would you send this reply to the customer as it stands?","The reply can be sent as written.","The reply needs changes before sending."),
        "de":("Würden Sie diese Antwort so an den Kunden senden?","Die Antwort kann unverändert gesendet werden.","Die Antwort muss vor dem Senden überarbeitet werden."),
    }[language]
    return Template.model_validate({"id":"customer-reply-approval","version":1,"language":language,"maxTokens":512,
        "questions":[{"id":"overall_approval","text":prompt,"labels":[{"id":"approved","description":positive},
            {"id":"rejected","description":negative}],"passLabels":["approved"]}]})


def custom_text_evaluation(language: str, prompt: str, positive_label: str, negative_label: str,
                           examples: list[dict] | None = None) -> Template:
    """One user-defined text judgment; instructions are not training data."""
    values = (prompt, positive_label, negative_label)
    if any(not isinstance(value, str) for value in values):
        raise ValueError("Custom task wording must be text")
    prompt, positive_label, negative_label = (unicodedata.normalize("NFC", value).strip() for value in values)
    normalized = (prompt, positive_label, negative_label)
    if any(unicodedata.category(char).startswith("C") or char in "\n\r\u2028\u2029"
           for value in normalized for char in value):
        raise ValueError("Custom task wording must be visible single-line text")
    units = lambda value: len(value.encode("utf-16-le")) // 2
    if not 1 <= units(prompt) <= 500 or any(not 1 <= units(value) <= 40 for value in (positive_label, negative_label)):
        raise ValueError("Custom question or answer length is invalid")
    if positive_label.lower() == negative_label.lower():
        raise ValueError("Custom answers must be distinct")
    return Template.model_validate({"id": "custom-text-evaluation", "version": 1, "language": language, "maxTokens": 512,
        "questions": [{"id": "judgment", "text": prompt, "labels": [
            {"id": "approved", "description": positive_label}, {"id": "rejected", "description": negative_label}],
            "passLabels": ["approved"], **({"examples": examples} if examples else {})}]})


def custom_text_seed(language: str) -> Template:
    try:
        prompt, positive, negative = {
            "en": ("Does this content meet the stated requirements?", "Yes", "No"),
            "de": ("Erfüllt dieser Inhalt die angegebenen Anforderungen?", "Ja", "Nein"),
        }[language]
    except KeyError:
        raise ValueError("Custom text evaluation supports English and German") from None
    return custom_text_evaluation(language, prompt, positive, negative)


def is_custom_text_template(template: Template) -> bool:
    try:
        question = template.questions[0]
        expected = custom_text_evaluation(template.language, question.text,
                                          question.labels[0].description, question.labels[1].description,
                                          [example.model_dump() for example in question.examples])
        return template == expected
    except (ValueError, IndexError):
        return False


def bundle_supports_template(bundle: dict, template: Template) -> bool:
    """The signed capability permits a shape, never AI use or a confidence claim.

    Callers must separately check the exact request digest against current rights.
    Trained/calibrated bundles cannot transfer their evidence to another rubric.
    """
    if template.language not in bundle["languages"] or template.maxTokens > bundle.get("max_tokens", 512):
        return False
    digest = commitment(template.model_dump(), "rateloop.evaluator.template.v1")
    if digest in bundle.get("template_commitments", []):
        return True
    return (bundle.get("task_capability") == CUSTOM_TEXT_CAPABILITY
            and not bundle.get("snapshot_id") and not bundle.get("calibrations")
            and is_custom_text_template(template))


def website_binary_question(template: Template):
    """Resolve the only supported website label mapping without guessing IDs."""
    if template == overall_approval(template.language) or is_custom_text_template(template):
        return template.questions[0]
    raise ValueError("Website jobs require a supported frozen binary text question")
