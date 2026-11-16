"""Shared result type for pipeline stages. See docs/CONTRACTS.md."""
from dataclasses import dataclass, field


@dataclass
class StageResult:
    ok: bool
    data: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    reason: str = ""
