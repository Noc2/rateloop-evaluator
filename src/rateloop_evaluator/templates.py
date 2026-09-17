"""Frozen website templates. Any wording change requires a new version."""
from .protocol import Template


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
