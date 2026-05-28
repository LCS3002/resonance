from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ScoreResult:
    score: float                    # 0-1 overall engagement
    region_scores: dict[str, float] # language, visual, prefrontal
    model: str                      # which model produced this
    latency_ms: float = 0.0


class BaseScorer(ABC):
    """Common interface for all neural engagement scorers."""

    @abstractmethod
    def score(self, text: str) -> ScoreResult:
        ...

    def is_available(self) -> bool:
        return True
