"""Golden path: every v0.6 capability chained end to end in one flow."""

import glob

import pytest
from click.testing import CliRunner
from qiskit import QuantumCircuit


@pytest.fixture(autouse=True)
def _data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("QBC_DATA_DIR", str(tmp_path))


def _ghz(n=4):
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.measure_all()
    return qc


CAL = sorted(glob.glob("tests/fixtures/calibration_snapshots/ibm_fez_*.json"))[-1]


def test_full_surface_end_to_end():
    from qb_compiler import (
        calibration_trend,
        make_receipt,
        rank_value,
        record_receipt,
        regression_check,
        verify_viability,
    )
    from qb_compiler.cost.shot_budget import shots_for_rate
    from qb_compiler.qiskit_plugin.transpiler_plugin import qb_transpile
    from qb_compiler.verify import accuracy_summary
    from qb_compiler.viability import check_viability
    from qb_compiler.windows import format_table

    qc = _ghz()

    # 1. preflight with error budget, band, calibration age
    via = check_viability(qc, backend="ibm_fez", n_seeds=2)
    assert via.error_budget and via.fidelity_typical_abs_error
    rendered = str(via)
    assert "Error budget" in rendered and "+-" in rendered

    # 2. best-of-N with the fidelity-scored path engaged
    _best, cands = qb_transpile(
        qc, backend="ibm_fez", calibration_path=CAL, n_seeds=3, return_candidates=True
    )
    assert len(cands) == 3 and all(0 < c["score"] <= 1 for c in cands)

    # 3. receipt -> baseline -> regression statuses
    r1 = make_receipt(qc, via, backend="ibm_fez", seed=0)
    record_receipt(r1)
    assert regression_check(r1).status in ("NO_BASELINE", "STABLE")
    import dataclasses

    degraded = dataclasses.replace(
        r1, predicted_fidelity=via.estimated_fidelity - 0.5, timestamp="2099-01-01T00:00:00+00:00"
    )
    record_receipt(degraded)
    assert regression_check(degraded).status == "REGRESSION"

    # 4. verify with a deterministic runner + accuracy record
    res = verify_viability(qc, lambda c, shots: {"0" * 4: shots - 3, "1" * 4: 3}, backend="ibm_fez")
    assert 0.95 < res.mirror_success <= 1.0
    summ = accuracy_summary()
    assert summ["n"] == 1

    # 5. value ranking + trend + table
    rows = rank_value(qc, ["ibm_fez"], n_seeds=1)
    assert rows and rows[0].fidelity_per_dollar
    assert calibration_trend("ibm_fez")[0] in ("degrading", "stable", "improving", "unknown")
    assert "ibm_fez" in format_table(rows)

    # 6. shot budgeting for the error rate the preflight predicted
    assert shots_for_rate(max(1 - via.estimated_fidelity, 0.01), rel_width=0.2) > 0


def test_qec_preflight_and_cli_smokes(tmp_path):
    pytest.importorskip("stim")
    from qb_compiler import qec_preflight

    pre = qec_preflight(3, 3, physical_error=0.01)
    assert 0 < pre.projected_ler < 1 and "LER" in str(pre)

    from qb_compiler.cli.main import cli

    qasm = (
        'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; '
        "h q[0]; cx q[0],q[1]; measure q -> c;"
    )
    f = tmp_path / "bell.qasm"
    f.write_text(qasm)
    runner = CliRunner()
    smokes = [
        ["preflight", str(f), "-b", "ibm_fez"],
        ["when", str(f), "--seeds", "1"],
        ["doctor"],
        ["info"],
    ]
    try:
        import qiskit_aer  # noqa: F401

        smokes.append(["verify", str(f), "-b", "ibm_fez", "--shots", "64"])
    except ImportError:
        pass
    for args in smokes:
        result = runner.invoke(cli, args)
        assert result.exit_code == 0, (args, result.output, result.exception)
