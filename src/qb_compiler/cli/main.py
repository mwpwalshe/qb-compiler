"""``qbc`` — command-line interface for qb-compiler.

Usage
-----
    qbc preflight circuit.qasm --backend ibm_fez
    qbc analyze circuit.qasm --backend ibm_fez
    qbc diff circuit.qasm --backend ibm_fez --vs ibm_torino
    qbc doctor
    qbc info
    qbc calibration show ibm_fez
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

try:
    import click
except ImportError:
    raise SystemExit(
        "The qb-compiler CLI requires 'click'. Install with: pip install click"
    ) from None

from qb_compiler._version import __version__
from qb_compiler.config import BACKEND_CONFIGS, get_backend_spec


@click.group()
@click.version_option(__version__, prog_name="qb-compiler")
def cli() -> None:
    """qb-compiler by QubitBoost — quantum execution intelligence toolkit."""


# ── qbc preflight ───────────────────────────────────────────────────


@cli.command()
@click.argument("circuit", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--backend",
    "-b",
    multiple=True,
    required=True,
    help="Target backend (e.g. ibm_fez). One backend only in open-source tier.",
)
@click.option("--seeds", "-n", default=2, type=int, help="Number of transpiler seeds.")
def preflight(circuit: str, backend: tuple[str, ...], seeds: int) -> None:
    """Quick viability check: VIABLE / CAUTION / DO NOT RUN."""
    if len(backend) > 1:
        click.echo("Multi-backend comparison requires QubitBoost.")
        click.echo("See https://qubitboost.io/compiler")
        sys.exit(1)

    backend_name = backend[0]
    qc = _load_qasm(circuit)

    from qb_compiler.viability import check_viability

    try:
        result = check_viability(qc, backend=backend_name, n_seeds=seeds)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Map status to user-facing labels
    label_map = {"VIABLE": "VIABLE", "MARGINAL": "CAUTION", "NOT VIABLE": "DO NOT RUN"}
    label = label_map.get(result.status, result.status)

    try:
        spec = get_backend_spec(backend_name)
    except Exception:
        spec = None

    click.echo()
    click.echo(f"  Circuit: {result.circuit_name}")
    click.echo(f"  Backend: {backend_name}" + (f" ({spec.n_qubits}q)" if spec else ""))
    click.echo()

    # Status line
    if label == "VIABLE" or label == "CAUTION":
        click.echo(f"  Status: {label}")
    else:
        click.echo(f"  Status: {label}")

    click.echo(f"  Estimated fidelity: {result.estimated_fidelity:.4f}")
    click.echo(f"  Depth: {result.depth}  (viable limit: {result.viable_depth})")
    click.echo(f"  2Q gates: {result.two_qubit_gate_count}")
    if result.cost_estimate_usd is not None:
        click.echo(f"  Cost (4096 shots): ${result.cost_estimate_usd:.4f}")
    click.echo()

    # Gate recommendations
    _show_gate_recommendations(qc, result.cost_estimate_usd)


# ── qbc analyze ─────────────────────────────────────────────────────


@cli.command()
@click.argument("circuit", type=click.Path(exists=True, dir_okay=False))
@click.option("--backend", "-b", required=True, help="Target backend (e.g. ibm_fez).")
@click.option("--seeds", "-n", default=2, type=int, help="Number of transpiler seeds.")
def analyze(circuit: str, backend: str, seeds: int) -> None:
    """Analyze circuit viability with suggestions for a single backend."""
    qc = _load_qasm(circuit)
    n_qubits = qc.num_qubits
    ops = qc.count_ops()
    total_gates = sum(ops.values())

    from qb_compiler.viability import check_viability

    try:
        result = check_viability(qc, backend=backend, n_seeds=seeds)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    try:
        spec = get_backend_spec(backend)
    except Exception:
        spec = None

    from qb_compiler.integrations.qubitboost import detect_circuit_type

    circuit_type, confidence = detect_circuit_type(qc)
    if circuit_type != "general":
        circuit_type_label = f"{circuit_type.upper()} ({confidence.value} confidence)"
    else:
        circuit_type_label = "General"

    click.echo()
    click.echo(f"  Circuit Analysis: {result.circuit_name}")
    click.echo(f"  Circuit type: {circuit_type_label}")
    click.echo(f"  Qubits: {n_qubits}  Gates: {total_gates}  Depth: {qc.depth()}")
    if ops:
        ops_str = ", ".join(f"{k}:{v}" for k, v in sorted(ops.items(), key=lambda x: -x[1])[:8])
        click.echo(f"  Gate breakdown: {ops_str}")
    click.echo()
    click.echo(f"  Backend: {backend}" + (f" ({spec.n_qubits}q)" if spec else ""))
    click.echo(f"  Status: {result.status}")
    click.echo(f"  Estimated fidelity: {result.estimated_fidelity:.4f}")
    click.echo(f"  Signal/noise ratio: {result.signal_to_noise:.1f}x")
    click.echo(f"  Depth: {result.depth}  (viable limit: {result.viable_depth})")
    click.echo(f"  2Q gates after transpilation: {result.two_qubit_gate_count}")
    if result.cost_estimate_usd is not None:
        click.echo(f"  Cost (4096 shots): ${result.cost_estimate_usd:.4f}")
    click.echo()
    click.echo(f"  {result.reason}")
    click.echo()

    if result.suggestions:
        click.echo("  Suggestions:")
        for s in result.suggestions:
            click.echo(f"    - {s}")
        click.echo()

    # Gate recommendations
    _show_gate_recommendations(qc, result.cost_estimate_usd)


# ── qbc diff ────────────────────────────────────────────────────────


@cli.command()
@click.argument("circuit", type=click.Path(exists=True, dir_okay=False))
@click.option("--backend", "-b", required=True, help="First backend (e.g. ibm_fez).")
@click.option("--vs", required=True, help="Second backend to compare (e.g. ibm_torino).")
@click.option("--seeds", "-n", default=2, type=int, help="Transpiler seeds per backend.")
def diff(circuit: str, backend: str, vs: str, seeds: int) -> None:
    """Compare circuit performance on two backends side by side."""
    qc = _load_qasm(circuit)

    from qb_compiler.viability import check_viability

    results = {}
    for bk in [backend, vs]:
        try:
            results[bk] = check_viability(qc, backend=bk, n_seeds=seeds)
        except Exception as e:
            click.echo(f"Error analyzing {bk}: {e}", err=True)
            sys.exit(1)

    r1 = results[backend]
    r2 = results[vs]

    click.echo()
    click.echo(f"  Circuit: {r1.circuit_name}")
    click.echo()

    # Side-by-side comparison
    hdr = f"  {'':24s} {'':>2s} {backend:>16s}   {vs:>16s}"
    click.echo(hdr)
    click.echo(f"  {'':24s} {'':>2s} {'-' * 16}   {'-' * 16}")

    def _row(label: str, v1: str, v2: str, better: int = 0) -> None:
        marker1 = " <" if better == 1 else ""
        marker2 = " <" if better == 2 else ""
        click.echo(f"  {label:24s}    {v1:>16s}{marker1:3s}{v2:>16s}{marker2}")

    _row("Status", r1.status, r2.status)
    b = 1 if r1.estimated_fidelity >= r2.estimated_fidelity else 2
    _row("Est. fidelity", f"{r1.estimated_fidelity:.4f}", f"{r2.estimated_fidelity:.4f}", b)
    b = 1 if r1.two_qubit_gate_count <= r2.two_qubit_gate_count else 2
    _row("2Q gates", str(r1.two_qubit_gate_count), str(r2.two_qubit_gate_count), b)
    b = 1 if r1.depth <= r2.depth else 2
    _row("Depth", str(r1.depth), str(r2.depth), b)
    _row("Viable depth", str(r1.viable_depth), str(r2.viable_depth))
    _row("SNR", f"{r1.signal_to_noise:.1f}x", f"{r2.signal_to_noise:.1f}x")

    c1 = f"${r1.cost_estimate_usd:.4f}" if r1.cost_estimate_usd else "N/A"
    c2 = f"${r2.cost_estimate_usd:.4f}" if r2.cost_estimate_usd else "N/A"
    if r1.cost_estimate_usd and r2.cost_estimate_usd:
        b = 1 if r1.cost_estimate_usd <= r2.cost_estimate_usd else 2
    else:
        b = 0
    _row("Cost/4096 shots", c1, c2, b)

    click.echo()

    # Winner
    if r1.estimated_fidelity > r2.estimated_fidelity:
        click.echo(
            f"  Recommendation: {backend} "
            f"(+{r1.estimated_fidelity - r2.estimated_fidelity:.4f} fidelity)"
        )
    elif r2.estimated_fidelity > r1.estimated_fidelity:
        click.echo(
            f"  Recommendation: {vs} "
            f"(+{r2.estimated_fidelity - r1.estimated_fidelity:.4f} fidelity)"
        )
    else:
        click.echo("  Backends are equivalent for this circuit.")

    click.echo()
    click.echo("  Live calibration comparison available at https://qubitboost.io/compiler")
    click.echo()


# ── qbc doctor ──────────────────────────────────────────────────────


@cli.command()
def doctor() -> None:
    """Check your quantum development environment."""
    try:
        from rich.console import Console
    except ImportError:
        click.echo("The 'rich' package is required for doctor. pip install rich")
        sys.exit(1)

    console = Console()
    console.print()
    console.print("[bold]qbc doctor[/bold]")
    console.print()

    all_ok = True

    # 1. qb-compiler version
    console.print(f"[green]\u2714[/green]  qb-compiler {__version__}")

    # 2. Python version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    console.print(f"[green]\u2714[/green]  Python {py_ver}")

    # 3. Qiskit
    try:
        import qiskit

        qver = qiskit.__version__
        if tuple(int(x) for x in qver.split(".")[:2]) >= (1, 0):
            console.print(f"[green]\u2714[/green]  Qiskit {qver}")
        else:
            console.print(f"[yellow]![/yellow]  Qiskit {qver} — recommend >=1.0")
            all_ok = False
    except ImportError:
        console.print("[red]\u2718[/red]  Qiskit not installed — pip install qiskit")
        all_ok = False

    # 4. IBM credentials
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService

        instances = QiskitRuntimeService.saved_accounts()
        if instances:
            console.print(
                f"[green]\u2714[/green]  IBM credentials configured ({len(instances)} account(s))"
            )
        else:
            console.print(
                "[yellow]![/yellow]  No IBM credentials saved — run: "
                "QiskitRuntimeService.save_account("
                "channel='ibm_quantum_platform', token='...')"
            )
    except ImportError:
        console.print(
            "[yellow]![/yellow]  qiskit-ibm-runtime not installed (needed for IBM backends)"
        )
    except Exception as e:
        console.print(f"[yellow]![/yellow]  IBM credentials check failed: {e}")

    # 5. Available backends
    n_backends = len(BACKEND_CONFIGS)
    console.print(f"[green]\u2714[/green]  {n_backends} backends configured")

    # 6. Calibration fixtures
    from pathlib import Path as _Path

    fixture_dir = (
        _Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "calibration_snapshots"
    )
    if not fixture_dir.exists():
        fixture_dir = _Path(__file__).resolve().parents[1] / "calibration" / "fixtures"
    fixtures = list(fixture_dir.glob("*.json")) if fixture_dir.exists() else []
    if fixtures:
        console.print(f"[green]\u2714[/green]  {len(fixtures)} calibration snapshot(s) available")
        for f in sorted(fixtures):
            console.print(f"         {f.name}")
    else:
        console.print("[yellow]![/yellow]  No calibration snapshots found")

    # 7. Optional: ML models
    try:
        from qb_compiler.ml import is_available

        if is_available():
            console.print("[green]\u2714[/green]  ML layout predictor available")
        else:
            console.print("[dim]-[/dim]  ML layout predictor not available (optional)")
    except ImportError:
        console.print("[dim]-[/dim]  ML layout predictor not available (optional)")

    # 8. QubitBoost SDK
    from qb_compiler.integrations.qubitboost import is_sdk_available

    if is_sdk_available():
        try:
            import qubitboost  # type: ignore[import-untyped]

            qb_ver = getattr(qubitboost, "__version__", "?")
            console.print(f"[green]\u2714[/green]  QubitBoost SDK {qb_ver}")
        except Exception:
            console.print("[green]\u2714[/green]  QubitBoost SDK available")
    else:
        console.print(
            "[dim]-[/dim]  QubitBoost SDK not installed (optional) — pip install qubitboost-sdk"
        )

    # 9. Core dependencies
    for pkg_name, import_name in [
        ("numpy", "numpy"),
        ("rustworkx", "rustworkx"),
        ("rich", "rich"),
    ]:
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "?")
            console.print(f"[green]\u2714[/green]  {pkg_name} {ver}")
        except ImportError:
            console.print(f"[red]\u2718[/red]  {pkg_name} not installed")
            all_ok = False

    console.print()
    if all_ok:
        console.print("[bold green]Environment looks good![/bold green]")
    else:
        console.print("[bold yellow]Some issues detected — see above.[/bold yellow]")
    console.print()


# ── qbc compile ──────────────────────────────────────────────────────


@cli.command()
@click.argument("circuit", type=click.Path(exists=True, dir_okay=False))
@click.option("--backend", "-b", default=None, help="Target backend (e.g. ibm_fez).")
@click.option("--output", "-o", default=None, type=click.Path(), help="Output file path.")
@click.option(
    "--strategy",
    "-s",
    type=click.Choice(["fidelity_optimal", "depth_optimal", "budget_optimal"]),
    default="fidelity_optimal",
    help="Compilation strategy.",
)
@click.option("--receipt", is_flag=True, help="Save compilation receipt as JSON.")
def compile(
    circuit: str,
    backend: str | None,
    output: str | None,
    strategy: str,
    receipt: bool,
) -> None:
    """Compile a QASM circuit file."""
    from qb_compiler.compiler import QBCircuit, QBCompiler

    path = Path(circuit)
    content = path.read_text()

    n_qubits = _parse_qasm_n_qubits(content)
    if n_qubits < 1:
        click.echo("Error: could not determine qubit count from QASM file.", err=True)
        sys.exit(1)

    qbc = QBCircuit(n_qubits)
    for gate_name, qubits, params in _parse_qasm_gates(content, n_qubits):
        qbc.add(gate_name, qubits, params)

    compiler = QBCompiler(backend=backend, strategy=strategy)
    result = compiler.compile(qbc)

    click.echo(
        f"Compiled: depth {result.original_depth} -> {result.compiled_depth} "
        f"({result.depth_reduction_pct:.1f}% reduction)"
    )
    click.echo(f"Estimated fidelity: {result.estimated_fidelity:.4f}")
    click.echo(f"Compilation time: {result.compilation_time_ms:.1f} ms")

    if output:
        Path(output).write_text(
            f"// Compiled by qb-compiler {__version__}\n"
            f"// Original depth: {result.original_depth}, "
            f"Compiled depth: {result.compiled_depth}\n"
            f"// Gates: {result.compiled_circuit.gate_count}\n"
        )
        click.echo(f"Written to {output}")

    if receipt:
        receipt_data = _build_receipt(result, path, backend, strategy)
        receipt_path = path.with_suffix(".receipt.json")
        import json

        receipt_path.write_text(json.dumps(receipt_data, indent=2))
        click.echo(f"Receipt saved to {receipt_path}")
        click.echo("View execution history and trends at https://qubitboost.io/dashboard")


# ── qbc info ─────────────────────────────────────────────────────────


@cli.command()
def info() -> None:
    """Show version and available backends."""
    click.echo(f"qb-compiler {__version__}")
    click.echo()
    click.echo("Available backends:")
    for name, spec in sorted(BACKEND_CONFIGS.items()):
        click.echo(
            f"  {name:20s}  {spec.provider:8s}  {spec.n_qubits:>4d}q  "
            f"${spec.cost_per_shot:.5f}/shot"
        )


# ── qbc calibration ──────────────────────────────────────────────────


@cli.group()
def calibration() -> None:
    """Calibration data commands."""


@calibration.command("show")
@click.argument("backend")
def calibration_show(backend: str) -> None:
    """Show latest calibration summary for BACKEND."""
    try:
        spec = get_backend_spec(backend)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    click.echo(f"Backend: {backend}")
    click.echo(f"Provider: {spec.provider}")
    click.echo(f"Qubits: {spec.n_qubits}")
    click.echo(f"Basis gates: {', '.join(spec.basis_gates)}")
    click.echo(f"Median CX error: {spec.median_cx_error:.4f}")
    click.echo(f"Median readout error: {spec.median_readout_error:.4f}")
    click.echo(f"Median T1: {spec.t1_us:.1f} us")
    click.echo(f"Median T2: {spec.t2_us:.1f} us")
    click.echo(f"Cost per shot: ${spec.cost_per_shot:.5f}")


# ── Helpers ──────────────────────────────────────────────────────────


def _show_gate_recommendations(qc: Any, cost_usd: float | None) -> None:
    """Show QubitBoost gate recommendations for a circuit."""
    from qb_compiler.integrations.qubitboost import (
        detect_circuit_type,
        is_sdk_available,
        recommend_gates,
    )

    circuit_type, confidence = detect_circuit_type(qc)
    recs = recommend_gates(circuit_type, confidence)
    if not recs:
        return

    click.echo("  QubitBoost gate eligibility:")
    for r in recs:
        click.echo(f"    * {r.gate:14s} {r.status} — {r.headline}")
        if r.validated_claim:
            click.echo(f"      {'':14s} Hardware-validated: {r.validated_claim} {r.qualifier}")
    click.echo()

    if is_sdk_available():
        click.echo("  QubitBoost SDK installed.")
    else:
        click.echo("  Requires: pip install qubitboost-sdk")
        click.echo("  Learn more: https://qubitboost.io")
    click.echo()


def _load_qasm(circuit_path: str) -> Any:
    """Load a QASM file as a Qiskit QuantumCircuit."""

    from qiskit import QuantumCircuit

    path = Path(circuit_path)
    try:
        return QuantumCircuit.from_qasm_file(str(path))
    except Exception as e:
        click.echo(f"Error loading circuit: {e}", err=True)
        sys.exit(1)


def _build_receipt(
    result: Any,
    circuit_path: Path,
    backend: str | None,
    strategy: str,
) -> dict[str, Any]:
    """Build a compilation receipt dict."""
    import datetime

    return {
        "qb_compiler_version": __version__,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "circuit_file": str(circuit_path),
        "backend": backend,
        "strategy": strategy,
        "original_depth": result.original_depth,
        "compiled_depth": result.compiled_depth,
        "depth_reduction_pct": round(result.depth_reduction_pct, 2),
        "estimated_fidelity": round(result.estimated_fidelity, 6),
        "compilation_time_ms": round(result.compilation_time_ms, 2),
        "gate_count": result.compiled_circuit.gate_count,
    }


def _parse_qasm_n_qubits(content: str) -> int:
    """Extract total qubit count from ``qreg`` declarations."""
    import re

    total = 0
    for m in re.finditer(r"qreg\s+\w+\[(\d+)\]", content):
        total += int(m.group(1))
    return total


def _parse_qasm_gates(
    content: str, n_qubits: int
) -> list[tuple[str, tuple[int, ...], tuple[float, ...]]]:
    """Minimal QASM 2.0 gate extractor."""
    import re

    gates: list[tuple[str, tuple[int, ...], tuple[float, ...]]] = []
    pattern = re.compile(
        r"^(\w+)"
        r"(?:\(([^)]*)\))?"
        r"\s+"
        r"([\w\[\],\s]+)"
        r"\s*;",
        re.MULTILINE,
    )
    for m in pattern.finditer(content):
        name = m.group(1).lower()
        if name in ("qreg", "creg", "include", "openqasm", "measure", "barrier"):
            continue
        param_str = m.group(2)
        params: tuple[float, ...] = ()
        if param_str:
            try:
                params = tuple(float(p.strip()) for p in param_str.split(","))
            except ValueError:
                params = ()
        qubit_str = m.group(3)
        qubit_refs = re.findall(r"\w+\[(\d+)\]", qubit_str)
        qubits = tuple(int(q) for q in qubit_refs)
        if qubits and all(0 <= q < n_qubits for q in qubits):
            gates.append((name, qubits, params))
    return gates
