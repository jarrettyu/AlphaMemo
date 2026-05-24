from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .math_utils import stable_sigmoid


@dataclass(slots=True)
class FactorNode:
    id: int
    formula: str
    ic: float
    icir: float
    quality: float
    category: str
    motif: str
    parent_id: int | None = None
    depth: int = 0
    times_selected: int = 0
    created_step: int = 0
    children: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class FactorDAG:
    """Graph state for parent retrieval in formula evolution."""

    def __init__(self, depth_decay: float = 0.05, times_decay: float = 0.10, start_times: int = 2) -> None:
        self.depth_decay = depth_decay
        self.times_decay = times_decay
        self.start_times = start_times
        self.nodes: list[FactorNode] = []
        self._formula_to_id: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.nodes)

    def add(
        self,
        formula: str,
        ic: float,
        icir: float,
        quality: float,
        category: str,
        motif: str,
        parent_id: int | None,
        step: int,
    ) -> int:
        if formula in self._formula_to_id:
            return self._formula_to_id[formula]
        depth = 0
        if parent_id is not None and 0 <= parent_id < len(self.nodes):
            depth = self.nodes[parent_id].depth + 1
        node_id = len(self.nodes)
        node = FactorNode(
            id=node_id,
            formula=formula,
            ic=ic,
            icir=icir,
            quality=quality,
            category=category,
            motif=motif,
            parent_id=parent_id,
            depth=depth,
            created_step=step,
        )
        self.nodes.append(node)
        self._formula_to_id[formula] = node_id
        if parent_id is not None and 0 <= parent_id < len(self.nodes):
            self.nodes[parent_id].children.append(node_id)
        return node_id

    def scores(self) -> np.ndarray:
        if not self.nodes:
            return np.asarray([], dtype=float)
        qualities = np.asarray([node.quality for node in self.nodes], dtype=float)
        standardized = (qualities - qualities.mean()) / (qualities.std() + 1e-6)
        quality_prob = stable_sigmoid(standardized)
        depths = np.asarray([node.depth for node in self.nodes], dtype=float)
        times = np.asarray([node.times_selected for node in self.nodes], dtype=float)
        depth_part = np.power(max(1.0 - self.depth_decay, 1e-6), depths)
        times_part = np.power(max(1.0 - self.times_decay, 1e-6), np.maximum(times - self.start_times, 0))
        return np.clip(quality_prob * depth_part * times_part, 1e-6, 1.0)

    def select(self, k: int) -> list[tuple[int, FactorNode, float]]:
        if not self.nodes:
            return []
        scores = self.scores()
        order = np.argsort(-scores)[:k]
        selected: list[tuple[int, FactorNode, float]] = []
        for idx in order:
            node = self.nodes[int(idx)]
            node.times_selected += 1
            selected.append((int(idx), node, float(scores[int(idx)])))
        return selected

    def best_parent_id(self) -> int | None:
        if not self.nodes:
            return None
        return int(np.argmax(self.scores()))

    def to_dict(self) -> dict:
        return {
            "depth_decay": self.depth_decay,
            "times_decay": self.times_decay,
            "start_times": self.start_times,
            "nodes": [node.to_dict() for node in self.nodes],
        }
