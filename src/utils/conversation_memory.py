from collections import deque


class ConversationMemory:
    def __init__(self, max_turns: int = 5) -> None:
        """Initialize a conversation memory buffer."""
        self.history = deque(maxlen=max_turns)

    def add_turn(self, user_query: str, model_answer: str) -> None:
        """Store a conversation turn."""
        self.history.append({"user": user_query, "assistant": model_answer})

    def get_context(self) -> str:
        """Return combined conversation context."""
        return " ".join(
            [
                f"User: {hist['user']} Assistant: {hist['assistant']}"
                for hist in self.history
            ]
        )

    def clear(self) -> None:
        """Clear all stored conversation history from memory."""
        self.history.clear()
