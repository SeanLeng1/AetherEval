def is_context_length_error(error: object) -> bool:
    """Return whether an inference failure is a deterministic context overflow."""

    text = str(error).casefold()
    if "bfcl prompt exceeds max context length" in text:
        return True

    mentions_input = any(
        phrase in text
        for phrase in (
            "input length",
            "prompt length",
            "the input (",
            "the prompt (",
        )
    )
    mentions_limit = any(
        phrase in text
        for phrase in (
            "exceeds the maximum allowed length",
            "exceeds max context length",
            "exceeds the model's context length",
            "longer than the model's context length",
            "longer than or equal to the model's context length",
        )
    )
    return mentions_input and mentions_limit
