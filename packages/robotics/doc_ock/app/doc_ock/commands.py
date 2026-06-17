from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass(frozen=True)
class TaskCommand:
    task_text: str


class CommandSource(Protocol):
    def next_command(self, timeout_s: Optional[float] = None) -> Optional[TaskCommand]:
        """Return the next command if one is available."""
