from abc import ABC, abstractmethod


class BaseNotifier(ABC):
    @abstractmethod
    async def notify(self, gate_id: str, context: dict) -> None:
        """Send an approval request. Resolves when the notification is sent,
        NOT when the human responds — that comes via approve()/deny()."""
