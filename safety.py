from logger import logger

# -----------------------
# BLOCKED WORDS
# -----------------------
UNSAFE_INPUT = ["hack", "illegal", "exploit"]

SENSITIVE_DATA = ["salary", "ssn", "confidential", "internal use only"]


# -----------------------
# INPUT GUARDRAILS
# -----------------------
def is_safe_input(question: str) -> bool:
    logger.info("Running input safety check")

    q = question.lower()
    safe = not any(word in q for word in UNSAFE_INPUT)

    if not safe:
        logger.warning("Blocked unsafe query")

    return safe


# -----------------------
# CONTEXT SANITIZATION (protecting sensitive data leakage to the API's)
# -----------------------
def sanitize_context(text: str) -> str:
    for word in SENSITIVE_DATA:
        text = text.replace(word, "[REDACTED]")
    return text