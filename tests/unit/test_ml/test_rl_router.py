"""Tests for RL SWAP router (Phase 4)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from qb_compiler.calibration.models.backend_properties import BackendProperties
from qb_compiler.calibration.models.coupling_properties import GateProperties
from qb_compiler.calibration.models.qubit_properties import QubitProperties
from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.operations import QBGate
from qb_compiler.ml.rl_router import (
    RL_HIDDEN_DIM,
    RL_STATE_DIM,
    RLRouter,
    RoutingAction,
    RoutingEnvironment,
    RoutingState,
    _build_ppo_model,
    _ppo_update,
    train_rl_router,
)


# ── fixtures ──────────────────────────────────────────────────────────


def _make_backend(n: int = 10) -> BackendProperties:
    qubits = [
        QubitProperties(
            qubit_id=i,
            t1_us=100.0 + i * 10,
            t2_us=80.0 + i * 5,
            readout_error=0.01 + i * 0.002,
            frequency_ghz=5.0 + i * 0.05,
        )
        for i in range(n)
    ]
    coupling = [(i, i + 1) for i in range(n - 1)] + [(i + 1, i) for i in range(n - 1)]
    gates = [
        GateProperties(gate_type="cx", qubits=(q0, q1), error_rate=0.005 + 0.001 * min(q0, q1))
        for q0, q1 in coupling
    ]
    return BackendProperties(
        backend="test",
        provider="test",
        n_qubits=n,
        basis_gates=("cx",),
        coupling_map=coupling,
        qubit_properties=qubits,
        gate_properties=gates,
        timestamp="2026-01-01",
    )


def _make_bell() -> QBCircuit:
    c = QBCircuit(n_qubits=2, n_clbits=0)
    c.add_gate(QBGate("h", (0,)))
    c.add_gate(QBGate("cx", (0, 1)))
    return c


def _make_ghz(n: int) -> QBCircuit:
    c = QBCircuit(n_qubits=n, n_clbits=0)
    c.add_gate(QBGate("h", (0,)))
    for i in range(n - 1):
        c.add_gate(QBGate("cx", (i, i + 1)))
    return c


def _make_nonlocal_circuit() -> QBCircuit:
    """Circuit requiring routing: qubits 0-3 on a 5-qubit chain."""
    c = QBCircuit(n_qubits=4, n_clbits=0)
    c.add_gate(QBGate("cx", (0, 1)))
    c.add_gate(QBGate("cx", (2, 3)))
    c.add_gate(QBGate("cx", (0, 3)))  # non-adjacent on chain
    return c


# ── environment tests ─────────────────────────────────────────────────


class TestRoutingEnvironment:
    def test_reset_returns_initial_state(self):
        backend = _make_backend(5)
        layout = {0: 0, 1: 1}
        env = RoutingEnvironment(backend, layout, _make_bell())
        state = env.reset()
        assert state.layout == {0: 0, 1: 1}
        assert state.accumulated_error == 0.0
        assert state.n_swaps == 0

    def test_layers_extracted(self):
        backend = _make_backend(5)
        layout = {i: i for i in range(4)}
        circ = _make_nonlocal_circuit()
        env = RoutingEnvironment(backend, layout, circ)
        state = env.reset()
        # Should have at least 2 layers (parallel cx(0,1)+cx(2,3), then cx(0,3))
        assert len(state.remaining_layers) >= 1

    def test_advance_reduces_layers(self):
        backend = _make_backend(5)
        layout = {0: 0, 1: 1}
        env = RoutingEnvironment(backend, layout, _make_bell())
        state = env.reset()
        n_layers = len(state.remaining_layers)

        action = RoutingAction(action_type="advance")
        new_state, reward, done = env.step(state, action)
        assert len(new_state.remaining_layers) == n_layers - 1

    def test_advance_on_executable_gate_has_small_error(self):
        backend = _make_backend(5)
        layout = {0: 0, 1: 1}  # Adjacent on chain
        env = RoutingEnvironment(backend, layout, _make_bell())
        state = env.reset()

        action = RoutingAction(action_type="advance")
        new_state, reward, done = env.step(state, action)
        # Reward should be negative (error) but small
        assert reward < 0
        assert reward > -0.1  # small error since qubits are adjacent

    def test_swap_updates_layout(self):
        backend = _make_backend(5)
        layout = {0: 0, 1: 1, 2: 2}
        env = RoutingEnvironment(backend, layout, _make_ghz(3))
        state = env.reset()

        action = RoutingAction(action_type="swap", swap_edge=(0, 1))
        new_state, reward, done = env.step(state, action)
        # Layout should be swapped
        assert new_state.layout[0] == 1
        assert new_state.layout[1] == 0
        assert new_state.n_swaps == 1

    def test_swap_has_3x_cx_error(self):
        backend = _make_backend(5)
        layout = {0: 0, 1: 1}
        env = RoutingEnvironment(backend, layout, _make_bell())
        state = env.reset()

        action = RoutingAction(action_type="swap", swap_edge=(0, 1))
        new_state, reward, done = env.step(state, action)
        # SWAP = 3 CX gates, each with error 0.005
        expected_error = 3 * 0.005
        assert abs(new_state.accumulated_error - expected_error) < 1e-6

    def test_done_when_all_layers_processed(self):
        backend = _make_backend(5)
        layout = {0: 0, 1: 1}
        env = RoutingEnvironment(backend, layout, _make_bell())
        state = env.reset()

        # Advance through all layers
        while not state.done:
            action = RoutingAction(action_type="advance")
            state, _, done = env.step(state, action)
            if done:
                break
        assert state.done

    def test_state_features_fixed_size(self):
        backend = _make_backend(5)
        layout = {0: 0, 1: 1}
        env = RoutingEnvironment(backend, layout, _make_bell())
        state = env.reset()
        features = env.state_features(state)
        assert len(features) == RL_STATE_DIM

    def test_action_space_size(self):
        backend = _make_backend(5)
        layout = {0: 0, 1: 1}
        env = RoutingEnvironment(backend, layout, _make_bell())
        # 4 edges (undirected) + 1 advance
        assert env.n_actions == 5


# ── PPO agent tests ───────────────────────────────────────────────────


class TestPPOAgent:
    def test_build(self):
        agent = _build_ppo_model(state_dim=RL_STATE_DIM, n_actions=10)
        assert agent is not None

    def test_forward(self):
        agent = _build_ppo_model(state_dim=RL_STATE_DIM, n_actions=10)
        state = torch.randn(1, RL_STATE_DIM)
        with torch.no_grad():
            logits, value = agent(state)
        assert logits.shape == (1, 10)
        assert value.shape == (1,)

    def test_get_action(self):
        agent = _build_ppo_model(state_dim=RL_STATE_DIM, n_actions=10)
        state = torch.randn(1, RL_STATE_DIM)
        action_idx, log_prob, value = agent.get_action(state)
        assert 0 <= action_idx < 10
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_evaluate(self):
        agent = _build_ppo_model(state_dim=RL_STATE_DIM, n_actions=10)
        states = torch.randn(5, RL_STATE_DIM)
        actions = torch.randint(0, 10, (5,))
        log_probs, values, entropy = agent.evaluate(states, actions)
        assert log_probs.shape == (5,)
        assert values.shape == (5,)
        assert entropy.shape == (5,)

    def test_parameter_count(self):
        agent = _build_ppo_model(state_dim=RL_STATE_DIM, n_actions=100)
        n_params = sum(p.numel() for p in agent.parameters())
        # Should be reasonable
        assert n_params < 100_000
        assert n_params > 1000

    def test_gradients_flow(self):
        agent = _build_ppo_model(state_dim=RL_STATE_DIM, n_actions=5)
        states = torch.randn(3, RL_STATE_DIM)
        actions = torch.randint(0, 5, (3,))
        log_probs, values, entropy = agent.evaluate(states, actions)

        loss = -log_probs.mean() + 0.5 * values.pow(2).mean()
        loss.backward()

        for name, param in agent.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


# ── PPO update tests ──────────────────────────────────────────────────


class TestPPOUpdate:
    def test_update_runs(self):
        from qb_compiler.ml.rl_router import RoutingStep

        agent = _build_ppo_model(state_dim=RL_STATE_DIM, n_actions=5)
        optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)

        trajectory = [
            RoutingStep(
                state_features=[0.0] * RL_STATE_DIM,
                action_idx=i % 5,
                reward=-0.01,
                log_prob=-1.5,
                value=0.0,
                done=(i == 4),
            )
            for i in range(5)
        ]

        # Should not raise
        _ppo_update(agent, optimizer, trajectory, n_epochs=2)

    def test_update_changes_parameters(self):
        from qb_compiler.ml.rl_router import RoutingStep

        agent = _build_ppo_model(state_dim=RL_STATE_DIM, n_actions=5)
        optimizer = torch.optim.Adam(agent.parameters(), lr=1e-2)

        # Record initial parameters
        initial_params = {
            n: p.data.clone() for n, p in agent.named_parameters()
        }

        trajectory = [
            RoutingStep(
                state_features=[float(i)] + [0.0] * (RL_STATE_DIM - 1),
                action_idx=i % 5,
                reward=-0.05 * (i + 1),
                log_prob=-1.5,
                value=0.0,
                done=(i == 9),
            )
            for i in range(10)
        ]

        _ppo_update(agent, optimizer, trajectory, n_epochs=5)

        # At least some parameters should have changed
        changed = False
        for n, p in agent.named_parameters():
            if not torch.allclose(p.data, initial_params[n]):
                changed = True
                break
        assert changed, "PPO update should change parameters"


# ── training tests ────────────────────────────────────────────────────


class TestRLTraining:
    def test_train_runs(self, tmp_path):
        backend = _make_backend(5)
        circuits = [_make_bell(), _make_ghz(3)]
        layouts = [{0: 0, 1: 1}, {0: 0, 1: 1, 2: 2}]

        output = tmp_path / "test_rl.pt"
        metadata = train_rl_router(
            backend=backend,
            circuits=circuits,
            initial_layouts=layouts,
            output_path=output,
            n_episodes=10,
            n_epochs_per_update=2,
            max_steps_per_episode=20,
            verbose=False,
        )

        assert output.exists()
        assert metadata["n_parameters"] > 0
        assert metadata["n_episodes"] == 10

    def test_train_produces_metadata(self, tmp_path):
        backend = _make_backend(5)
        circuits = [_make_bell()]
        layouts = [{0: 0, 1: 1}]

        output = tmp_path / "test_rl.pt"
        metadata = train_rl_router(
            backend=backend,
            circuits=circuits,
            initial_layouts=layouts,
            output_path=output,
            n_episodes=5,
            verbose=False,
        )

        meta_path = output.with_suffix(".meta.json")
        assert meta_path.exists()
        assert metadata["architecture"] == "ppo_actor_critic"
        assert metadata["backend"] == "test"

    def test_empty_circuits_raises(self, tmp_path):
        backend = _make_backend(5)
        with pytest.raises(ValueError, match="at least one"):
            train_rl_router(
                backend=backend,
                circuits=[],
                initial_layouts=[],
                output_path=tmp_path / "test.pt",
                verbose=False,
            )


# ── router inference tests ────────────────────────────────────────────


class TestRLRouter:
    @pytest.fixture()
    def trained_router(self, tmp_path) -> RLRouter:
        backend = _make_backend(5)
        circuits = [_make_bell(), _make_ghz(3)]
        layouts = [{0: 0, 1: 1}, {0: 0, 1: 1, 2: 2}]

        output = tmp_path / "test_rl.pt"
        train_rl_router(
            backend=backend,
            circuits=circuits,
            initial_layouts=layouts,
            output_path=output,
            n_episodes=10,
            verbose=False,
        )

        return RLRouter(model_path=output, backend=backend)

    def test_route_bell(self, trained_router: RLRouter):
        layout = {0: 0, 1: 1}
        final_layout, swaps, error = trained_router.route(_make_bell(), layout)
        assert isinstance(final_layout, dict)
        assert isinstance(swaps, list)
        assert error >= 0

    def test_route_ghz(self, trained_router: RLRouter):
        layout = {0: 0, 1: 1, 2: 2}
        final_layout, swaps, error = trained_router.route(_make_ghz(3), layout)
        assert len(final_layout) >= 3

    def test_missing_weights_raises(self):
        backend = _make_backend(5)
        with pytest.raises(FileNotFoundError, match="not found"):
            RLRouter(model_path="/nonexistent/model.pt", backend=backend)

    def test_metadata(self, trained_router: RLRouter):
        meta = trained_router.metadata
        assert isinstance(meta, dict)
        assert "version" in meta
