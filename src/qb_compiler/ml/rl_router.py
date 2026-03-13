"""RL-based SWAP routing using PPO (Phase 4).

Reinforcement learning agent that learns optimal SWAP insertion
decisions during circuit routing.  The agent observes the current
routing state (partially routed circuit + device calibration) and
decides where to insert SWAPs to minimise accumulated gate error.

Architecture:
    State  = (current_layer_features, device_graph_embedding, routing_progress)
    Action = (insert SWAP on edge (i,j)) or (advance to next layer)
    Reward = negative calibration-aware error of inserted gates

The RL agent is trained per-backend using nightly calibration snapshots.
This module is **proprietary** — it requires ``qubitboost-sdk`` for
production models.  The open-source version provides the agent
architecture and training loop for research/educational use.

Requires: ``pip install "qb-compiler[gnn]"`` (uses PyTorch)
"""

from __future__ import annotations

import json
import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qb_compiler.calibration.models.backend_properties import BackendProperties
    from qb_compiler.ir.circuit import QBCircuit

logger = logging.getLogger(__name__)

_WEIGHTS_DIR = Path(__file__).parent / "_weights"

# ── Constants ─────────────────────────────────────────────────────────

RL_STATE_DIM = 64      # state embedding dimension
RL_HIDDEN_DIM = 128    # policy/value network hidden dim
MAX_SWAPS_PER_LAYER = 10  # safety limit


def _check_torch() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for RL routing. "
            "Install with: pip install 'qb-compiler[gnn]'"
        ) from exc


# ── Routing Environment ──────────────────────────────────────────────


@dataclass
class RoutingState:
    """Observable state of the routing environment."""

    # Current logical-to-physical qubit mapping
    layout: dict[int, int]
    # Remaining layers to route (each layer = list of 2Q gate pairs)
    remaining_layers: list[list[tuple[int, int]]]
    # Accumulated error so far
    accumulated_error: float
    # Number of SWAPs inserted
    n_swaps: int
    # Device adjacency (physical)
    adjacency: dict[int, set[int]]
    # Gate errors per physical edge
    edge_errors: dict[tuple[int, int], float]

    @property
    def done(self) -> bool:
        return len(self.remaining_layers) == 0

    @property
    def current_layer(self) -> list[tuple[int, int]] | None:
        return self.remaining_layers[0] if self.remaining_layers else None


@dataclass
class RoutingAction:
    """An action in the routing environment."""

    action_type: str  # "swap" or "advance"
    swap_edge: tuple[int, int] | None = None  # physical qubits to swap


@dataclass
class RoutingStep:
    """A single step in a routing episode."""

    state_features: list[float]
    action_idx: int
    reward: float
    log_prob: float
    value: float
    done: bool


class RoutingEnvironment:
    """Routing environment for RL training.

    Converts a circuit + layout into a sequential decision problem:
    for each layer of 2Q gates, the agent can insert SWAPs to make
    gates executable, then advance to execute the layer.

    Parameters
    ----------
    backend :
        Backend calibration data.
    initial_layout :
        Starting logical-to-physical qubit mapping.
    circuit :
        Circuit to route.
    """

    def __init__(
        self,
        backend: BackendProperties,
        initial_layout: dict[int, int],
        circuit: QBCircuit,
    ) -> None:
        self._backend = backend
        self._initial_layout = dict(initial_layout)

        # Build adjacency and error maps
        self._adjacency: dict[int, set[int]] = defaultdict(set)
        self._edge_errors: dict[tuple[int, int], float] = {}
        for gp in backend.gate_properties:
            if len(gp.qubits) == 2 and gp.error_rate is not None:
                q0, q1 = gp.qubits
                self._adjacency[q0].add(q1)
                self._adjacency[q1].add(q0)
                self._edge_errors[(q0, q1)] = gp.error_rate
                self._edge_errors[(q1, q0)] = gp.error_rate

        # Extract circuit layers
        self._layers = self._extract_layers(circuit)

        # Available SWAP edges (all coupling map edges)
        self._swap_edges: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for q0, q1 in backend.coupling_map:
            edge = (min(q0, q1), max(q0, q1))
            if edge not in seen:
                seen.add(edge)
                self._swap_edges.append(edge)

        # Action space: n_edges (SWAP) + 1 (advance)
        self.n_actions = len(self._swap_edges) + 1
        self._advance_action = len(self._swap_edges)

    def _extract_layers(self, circuit: QBCircuit) -> list[list[tuple[int, int]]]:
        """Extract 2Q gate layers from the circuit."""
        from qb_compiler.ir.operations import QBGate

        # Simple layering: greedily assign gates to layers
        layers: list[list[tuple[int, int]]] = []
        qubit_layer: dict[int, int] = {}

        for op in circuit.iter_ops():
            if isinstance(op, QBGate) and op.num_qubits >= 2:
                q0, q1 = op.qubits[0], op.qubits[1]
                layer_idx = max(qubit_layer.get(q0, 0), qubit_layer.get(q1, 0))
                while len(layers) <= layer_idx:
                    layers.append([])
                layers[layer_idx].append((q0, q1))
                qubit_layer[q0] = layer_idx + 1
                qubit_layer[q1] = layer_idx + 1

        return layers

    def reset(self) -> RoutingState:
        """Reset environment to initial state."""
        return RoutingState(
            layout=dict(self._initial_layout),
            remaining_layers=[list(layer) for layer in self._layers],
            accumulated_error=0.0,
            n_swaps=0,
            adjacency=dict(self._adjacency),
            edge_errors=dict(self._edge_errors),
        )

    def step(
        self, state: RoutingState, action: RoutingAction
    ) -> tuple[RoutingState, float, bool]:
        """Execute one action and return (new_state, reward, done).

        Reward is negative error (higher reward = less error).
        """
        if action.action_type == "advance":
            return self._advance(state)
        elif action.action_type == "swap" and action.swap_edge is not None:
            return self._insert_swap(state, action.swap_edge)
        else:
            # Invalid action — small penalty
            return state, -0.01, False

    def _advance(
        self, state: RoutingState
    ) -> tuple[RoutingState, float, bool]:
        """Execute the current layer and advance."""
        if not state.remaining_layers:
            return state, 0.0, True

        layer = state.remaining_layers[0]
        error = 0.0

        # Execute each 2Q gate in the layer
        for log_q0, log_q1 in layer:
            phys_q0 = state.layout.get(log_q0, log_q0)
            phys_q1 = state.layout.get(log_q1, log_q1)

            # Check if gate is executable (qubits are adjacent)
            if phys_q1 in state.adjacency.get(phys_q0, set()):
                # Gate is executable — add its error
                gate_err = state.edge_errors.get(
                    (phys_q0, phys_q1),
                    state.edge_errors.get((phys_q1, phys_q0), 0.01),
                )
                error += gate_err
            else:
                # Gate NOT executable — penalty for unresolved routing
                error += 0.1  # large penalty

        new_state = RoutingState(
            layout=dict(state.layout),
            remaining_layers=state.remaining_layers[1:],
            accumulated_error=state.accumulated_error + error,
            n_swaps=state.n_swaps,
            adjacency=state.adjacency,
            edge_errors=state.edge_errors,
        )

        done = len(new_state.remaining_layers) == 0
        reward = -error  # negative error = reward
        return new_state, reward, done

    def _insert_swap(
        self, state: RoutingState, edge: tuple[int, int]
    ) -> tuple[RoutingState, float, bool]:
        """Insert a SWAP gate on the given physical edge."""
        phys_a, phys_b = edge

        # SWAP error = 3 CX gates worth of error
        cx_err = state.edge_errors.get(
            (phys_a, phys_b),
            state.edge_errors.get((phys_b, phys_a), 0.01),
        )
        swap_error = 3 * cx_err

        # Update layout: find which logical qubits are on these physical qubits
        new_layout = dict(state.layout)
        log_a = None
        log_b = None
        for log_q, phys_q in new_layout.items():
            if phys_q == phys_a:
                log_a = log_q
            elif phys_q == phys_b:
                log_b = log_q

        if log_a is not None and log_b is not None:
            new_layout[log_a] = phys_b
            new_layout[log_b] = phys_a
        elif log_a is not None:
            new_layout[log_a] = phys_b
        elif log_b is not None:
            new_layout[log_b] = phys_a

        new_state = RoutingState(
            layout=new_layout,
            remaining_layers=state.remaining_layers,
            accumulated_error=state.accumulated_error + swap_error,
            n_swaps=state.n_swaps + 1,
            adjacency=state.adjacency,
            edge_errors=state.edge_errors,
        )

        reward = -swap_error
        return new_state, reward, False

    def state_features(self, state: RoutingState) -> list[float]:
        """Extract fixed-size feature vector from routing state."""
        features: list[float] = []

        # Progress features
        total_layers = len(self._layers)
        remaining = len(state.remaining_layers)
        features.append(remaining / max(total_layers, 1))
        features.append(state.accumulated_error)
        features.append(state.n_swaps / max(total_layers, 1))

        # Current layer features
        if state.current_layer:
            features.append(len(state.current_layer) / 10.0)
            # How many gates in current layer are executable?
            n_exec = 0
            for log_q0, log_q1 in state.current_layer:
                phys_q0 = state.layout.get(log_q0, log_q0)
                phys_q1 = state.layout.get(log_q1, log_q1)
                if phys_q1 in state.adjacency.get(phys_q0, set()):
                    n_exec += 1
            features.append(n_exec / max(len(state.current_layer), 1))
        else:
            features.extend([0.0, 1.0])

        # Pad to fixed size
        while len(features) < RL_STATE_DIM:
            features.append(0.0)

        return features[:RL_STATE_DIM]

    def action_to_routing_action(self, action_idx: int) -> RoutingAction:
        """Convert action index to RoutingAction."""
        if action_idx == self._advance_action:
            return RoutingAction(action_type="advance")
        elif 0 <= action_idx < len(self._swap_edges):
            return RoutingAction(
                action_type="swap",
                swap_edge=self._swap_edges[action_idx],
            )
        else:
            return RoutingAction(action_type="advance")


# ── PPO Agent ─────────────────────────────────────────────────────────


def _build_ppo_model(
    state_dim: int = RL_STATE_DIM,
    n_actions: int = 1,
    hidden_dim: int = RL_HIDDEN_DIM,
) -> "PPOAgent":
    """Build the PPO policy+value network."""
    _check_torch()
    import torch
    import torch.nn as nn

    class PPOAgent(nn.Module):
        """Actor-Critic PPO agent for SWAP routing."""

        def __init__(self) -> None:
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.policy_head = nn.Linear(hidden_dim, n_actions)
            self.value_head = nn.Linear(hidden_dim, 1)

        def forward(
            self, state: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Return (action_logits, state_value)."""
            h = self.shared(state)
            logits = self.policy_head(h)
            value = self.value_head(h)
            return logits, value.squeeze(-1)

        def get_action(
            self, state: torch.Tensor
        ) -> tuple[int, float, float]:
            """Sample an action and return (action_idx, log_prob, value)."""
            logits, value = self(state)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            return action.item(), dist.log_prob(action).item(), value.item()

        def evaluate(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Evaluate actions for PPO update."""
            logits, values = self(states)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            return log_probs, values, entropy

    return PPOAgent()


# ── RL Router (inference) ─────────────────────────────────────────────


class RLRouter:
    """RL-based SWAP router using a trained PPO agent.

    Routes circuits by sequentially deciding SWAP insertions for each
    layer, using calibration-aware error as the optimization target.

    Parameters
    ----------
    model_path :
        Path to saved PPO agent weights.
    backend :
        Backend calibration data.
    max_swaps_per_layer :
        Safety limit on SWAPs per layer to prevent infinite loops.
    """

    def __init__(
        self,
        model_path: str | Path,
        backend: BackendProperties,
        max_swaps_per_layer: int = MAX_SWAPS_PER_LAYER,
    ) -> None:
        _check_torch()
        import torch

        self._backend = backend
        self._max_swaps = max_swaps_per_layer

        self._model_path = Path(model_path)
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"RL router weights not found at {self._model_path}. "
                f"Train with: python -m qb_compiler.ml.train_rl"
            )

        # We need to know n_actions to build the model
        # Count coupling map edges
        seen: set[tuple[int, int]] = set()
        for q0, q1 in backend.coupling_map:
            edge = (min(q0, q1), max(q0, q1))
            seen.add(edge)
        n_actions = len(seen) + 1  # edges + advance

        self._agent = _build_ppo_model(
            state_dim=RL_STATE_DIM,
            n_actions=n_actions,
            hidden_dim=RL_HIDDEN_DIM,
        )
        state = torch.load(str(self._model_path), map_location="cpu", weights_only=True)
        self._agent.load_state_dict(state)
        self._agent.eval()

        # Load metadata
        meta_path = self._model_path.with_suffix(".meta.json")
        self._metadata: dict = {}
        if meta_path.exists():
            with open(meta_path) as f:
                self._metadata = json.load(f)

        logger.info(
            "Loaded RL router from %s (version=%s)",
            self._model_path.name,
            self._metadata.get("version", "unknown"),
        )

    def route(
        self,
        circuit: QBCircuit,
        initial_layout: dict[int, int],
    ) -> tuple[dict[int, int], list[tuple[int, int]], float]:
        """Route a circuit using the trained RL agent.

        Returns
        -------
        final_layout :
            The qubit mapping after routing.
        swaps :
            List of (phys_a, phys_b) SWAP insertions.
        total_error :
            Accumulated routing error estimate.
        """
        import torch

        env = RoutingEnvironment(self._backend, initial_layout, circuit)
        state = env.reset()
        swaps: list[tuple[int, int]] = []
        layer_swap_count = 0

        while not state.done:
            features = env.state_features(state)
            state_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                action_idx, _, _ = self._agent.get_action(state_tensor)

            action = env.action_to_routing_action(action_idx)

            if action.action_type == "swap" and action.swap_edge is not None:
                if layer_swap_count >= self._max_swaps:
                    # Force advance if too many SWAPs
                    action = RoutingAction(action_type="advance")
                    layer_swap_count = 0
                else:
                    swaps.append(action.swap_edge)
                    layer_swap_count += 1

            if action.action_type == "advance":
                layer_swap_count = 0

            state, _, done = env.step(state, action)
            if done:
                break

        return state.layout, swaps, state.accumulated_error

    @property
    def metadata(self) -> dict:
        return dict(self._metadata)


# ── Training ──────────────────────────────────────────────────────────


def train_rl_router(
    backend: BackendProperties,
    circuits: list[QBCircuit],
    initial_layouts: list[dict[int, int]],
    output_path: str | Path | None = None,
    n_episodes: int = 100,
    n_epochs_per_update: int = 4,
    lr: float = 3e-4,
    gamma: float = 0.99,
    clip_eps: float = 0.2,
    entropy_coeff: float = 0.01,
    max_steps_per_episode: int = 200,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Train PPO agent for SWAP routing on a specific backend.

    Parameters
    ----------
    backend :
        Backend calibration data.
    circuits :
        Training circuits.
    initial_layouts :
        Pre-computed initial layouts for each circuit.
    output_path :
        Where to save trained weights.
    n_episodes :
        Number of training episodes.
    n_epochs_per_update :
        PPO epochs per batch of experience.
    lr :
        Learning rate.
    gamma :
        Discount factor.
    clip_eps :
        PPO clipping parameter.
    entropy_coeff :
        Entropy bonus coefficient.
    max_steps_per_episode :
        Safety limit per episode.
    seed :
        Random seed.
    verbose :
        Print progress.

    Returns
    -------
    dict
        Training metadata.
    """
    _check_torch()
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    random.seed(seed)

    if output_path is None:
        _WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = _WEIGHTS_DIR / "rl_router_v1.pt"
    output_path = Path(output_path)

    if not circuits or not initial_layouts:
        raise ValueError("Need at least one circuit and layout for training")

    # Build environment from first circuit to get action space size
    env0 = RoutingEnvironment(backend, initial_layouts[0], circuits[0])
    n_actions = env0.n_actions

    agent = _build_ppo_model(
        state_dim=RL_STATE_DIM,
        n_actions=n_actions,
        hidden_dim=RL_HIDDEN_DIM,
    )
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)

    episode_rewards: list[float] = []
    episode_swaps: list[int] = []

    for episode in range(n_episodes):
        # Pick a random circuit
        idx = random.randint(0, len(circuits) - 1)
        circuit = circuits[idx]
        layout = initial_layouts[idx]

        env = RoutingEnvironment(backend, layout, circuit)
        state = env.reset()

        trajectory: list[RoutingStep] = []
        total_reward = 0.0
        n_swaps_ep = 0

        for step_i in range(max_steps_per_episode):
            if state.done:
                break

            features = env.state_features(state)
            state_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

            action_idx, log_prob, value = agent.get_action(state_tensor)
            action = env.action_to_routing_action(action_idx)

            state, reward, done = env.step(state, action)
            total_reward += reward

            if action.action_type == "swap":
                n_swaps_ep += 1

            trajectory.append(RoutingStep(
                state_features=features,
                action_idx=action_idx,
                reward=reward,
                log_prob=log_prob,
                value=value,
                done=done,
            ))

            if done:
                break

        episode_rewards.append(total_reward)
        episode_swaps.append(n_swaps_ep)

        # PPO update
        if trajectory:
            _ppo_update(
                agent, optimizer, trajectory,
                gamma=gamma, clip_eps=clip_eps,
                entropy_coeff=entropy_coeff,
                n_epochs=n_epochs_per_update,
            )

        if verbose and (episode + 1) % 20 == 0:
            recent_r = sum(episode_rewards[-20:]) / min(20, len(episode_rewards[-20:]))
            recent_s = sum(episode_swaps[-20:]) / min(20, len(episode_swaps[-20:]))
            print(
                f"  Episode {episode + 1}/{n_episodes}: "
                f"avg_reward={recent_r:.4f}, avg_swaps={recent_s:.1f}"
            )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent.state_dict(), str(output_path))

    n_params = sum(p.numel() for p in agent.parameters())
    model_size = output_path.stat().st_size

    metadata = {
        "version": "1.0.0",
        "architecture": "ppo_actor_critic",
        "state_dim": RL_STATE_DIM,
        "hidden_dim": RL_HIDDEN_DIM,
        "n_actions": n_actions,
        "n_parameters": n_params,
        "model_size_bytes": model_size,
        "n_episodes": n_episodes,
        "final_avg_reward": sum(episode_rewards[-20:]) / max(1, min(20, len(episode_rewards))),
        "final_avg_swaps": sum(episode_swaps[-20:]) / max(1, min(20, len(episode_swaps))),
        "backend": backend.backend,
    }

    meta_path = output_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f"\nRL router saved to {output_path}")
        print(f"  Parameters: {n_params:,}")
        print(f"  Size: {model_size / 1024:.1f} KB")
        print(f"  Final avg reward: {metadata['final_avg_reward']:.4f}")

    return metadata


def _ppo_update(
    agent,
    optimizer,
    trajectory: list[RoutingStep],
    gamma: float = 0.99,
    clip_eps: float = 0.2,
    entropy_coeff: float = 0.01,
    n_epochs: int = 4,
) -> None:
    """Perform PPO update on a batch of experience."""
    import torch

    # Compute returns
    returns: list[float] = []
    R = 0.0
    for step in reversed(trajectory):
        R = step.reward + gamma * R * (0.0 if step.done else 1.0)
        returns.insert(0, R)

    states = torch.tensor(
        [s.state_features for s in trajectory], dtype=torch.float32
    )
    actions = torch.tensor(
        [s.action_idx for s in trajectory], dtype=torch.long
    )
    old_log_probs = torch.tensor(
        [s.log_prob for s in trajectory], dtype=torch.float32
    )
    returns_t = torch.tensor(returns, dtype=torch.float32)

    # Normalise returns
    if returns_t.std() > 1e-8:
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

    for _ in range(n_epochs):
        log_probs, values, entropy = agent.evaluate(states, actions)
        advantages = returns_t - values.detach()

        # PPO clipped objective
        ratio = torch.exp(log_probs - old_log_probs.detach())
        clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
        policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()

        # Value loss
        value_loss = 0.5 * (returns_t - values).pow(2).mean()

        # Entropy bonus
        entropy_loss = -entropy_coeff * entropy.mean()

        loss = policy_loss + value_loss + entropy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
        optimizer.step()
