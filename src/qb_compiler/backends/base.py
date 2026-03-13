"""Backend target description for the compiler."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BackendTarget:
    """Hardware target that the compiler compiles *to*.

    Encapsulates the physical constraints — qubit count, native gate set,
    and coupling topology — that compilation passes must respect.

    Parameters
    ----------
    n_qubits:
        Number of physical qubits.
    basis_gates:
        Tuple of native gate names the backend can execute.
    coupling_map:
        Directed adjacency list ``[(ctrl, tgt), ...]``.  An empty list
        means all-to-all connectivity.
    name:
        Optional human-readable name.
    """

    n_qubits: int
    basis_gates: tuple[str, ...]
    coupling_map: list[tuple[int, int]] = field(default_factory=list)
    name: str = "unknown"

    def __post_init__(self) -> None:
        # Pre-compute adjacency for BFS distance queries
        object.__setattr__(self, "_adjacency", self._build_adjacency())

    # ── gate support ─────────────────────────────────────────────────

    def supports_gate(self, gate: str) -> bool:
        """Check whether *gate* is in the native gate set."""
        return gate in self.basis_gates

    # ── topology ─────────────────────────────────────────────────────

    @property
    def is_fully_connected(self) -> bool:
        """True if coupling_map is empty (all-to-all assumed)."""
        return len(self.coupling_map) == 0

    def are_connected(self, q1: int, q2: int) -> bool:
        """Check whether qubits *q1* and *q2* share a direct coupling."""
        if self.is_fully_connected:
            return True
        adj = self._get_adjacency()
        return q2 in adj.get(q1, set())

    def qubit_distance(self, q1: int, q2: int) -> int:
        """Shortest-path distance between *q1* and *q2* on the coupling graph.

        Returns 0 if ``q1 == q2``, 1 if directly coupled, etc.
        For all-to-all connectivity, always returns ``min(1, |q1 - q2|)``.
        Raises ``ValueError`` if no path exists.
        """
        if q1 == q2:
            return 0
        if self.is_fully_connected:
            return 1

        adj = self._get_adjacency()
        # BFS
        visited: set[int] = {q1}
        queue: deque[tuple[int, int]] = deque([(q1, 0)])
        while queue:
            node, dist = queue.popleft()
            for neighbour in adj.get(node, set()):
                if neighbour == q2:
                    return dist + 1
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append((neighbour, dist + 1))

        raise ValueError(
            f"No path between qubit {q1} and qubit {q2} in coupling graph"
        )

    def neighbours(self, qubit: int) -> frozenset[int]:
        """Return the set of qubits directly coupled to *qubit*."""
        if self.is_fully_connected:
            return frozenset(range(self.n_qubits)) - {qubit}
        adj = self._get_adjacency()
        return frozenset(adj.get(qubit, set()))

    # ── internals ────────────────────────────────────────────────────

    def _build_adjacency(self) -> dict[int, set[int]]:
        """Build undirected adjacency dict from the coupling map."""
        adj: dict[int, set[int]] = {}
        for q1, q2 in self.coupling_map:
            adj.setdefault(q1, set()).add(q2)
            adj.setdefault(q2, set()).add(q1)
        return adj

    def _get_adjacency(self) -> dict[int, set[int]]:
        return object.__getattribute__(self, "_adjacency")  # type: ignore[no-any-return]

    # ── factory ──────────────────────────────────────────────────────

    @classmethod
    def from_backend_properties(
        cls,
        props: BackendProperties,  # type: ignore[name-defined]  # noqa: F821
    ) -> BackendTarget:
        """Build from a :class:`BackendProperties` calibration snapshot."""
        return cls(
            n_qubits=props.n_qubits,
            basis_gates=props.basis_gates,
            coupling_map=props.coupling_map,
            name=props.backend,
        )

    def __repr__(self) -> str:
        edges = len(self.coupling_map)
        return (
            f"BackendTarget(name={self.name!r}, n_qubits={self.n_qubits}, "
            f"basis_gates={self.basis_gates}, edges={edges})"
        )
