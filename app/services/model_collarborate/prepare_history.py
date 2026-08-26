import logging

logger = logging.getLogger(__name__)


def prepare_history(recent_messages: list | None, summary: str | None) -> str:
    """
    Formats the conversation summary and recent messages into one history section for prompts.

    Parameters:
    - recent_messages (list | None): messages since the last summary — comes from the caller (PromptBuilder, ResponseGate)
    - summary (str | None): the conversation's running summary — comes from the caller

    Returns:
    - str: the formatted history text — goes to the caller for use in a prompt
    """
    if recent_messages is None:
        logger.warning("prepare_history received recent_messages=None — treating as no recent messages.")
        recent_messages = []

    summary_part = f"Summary of earlier conversation:\n{summary}" if summary else None
    recent_part = (
        "\n".join(
            f"{'User' if m.sender == 'user' else 'Assistant'}: {m.text}"
            for m in recent_messages
        )
        if recent_messages
        else None
    )

    if summary_part and recent_part:
        return f"{summary_part}\n\nMore recent messages since that summary:\n{recent_part}"

    if summary_part:
        return summary_part

    if recent_part:
        return recent_part

    return "This is a brand new conversation. No recent conversation history available."
