# safety.py

DISALLOWED_KEYWORDS = [
    "build a bomb",
    "make a weapon",
    "kill myself",
    "suicide",
    "hack into",
    "credit card number"
    # add more
]

REFUSAL_MESSAGE = (
    "I'm sorry, but I can't help with that request. "
    "If you're in danger or thinking about self-harm, please reach out to a trusted person "
    "or a professional for support."
)

def is_unsafe(prompt: str) -> bool:
    lower = prompt.lower()
    return any(keyword in lower for keyword in DISALLOWED_KEYWORDS)
