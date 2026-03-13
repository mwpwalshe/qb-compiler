"""``qbc`` — command-line interface for qb-compiler.

Usage
-----
    qbc compile circuit.qasm --backend ibm_fez --output compiled.qasm
    qbc info
    qbc calibration show ibm_fez
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import click
except ImportError:
    # click is not a hard dependency — fail gracefully at import time
    # so the rest of the package stays importable.
    raise SystemExit(
        "The qb-compiler CLI requires 'click'. Install with: pip install click"
    ) from None

from qb_compiler._version import __version__
from qb_compiler.config import BACKEND_CONFIGS, get_backend_spec


@click.group()
@click.version_option(__version__, prog_name="qb-compiler")
def cli() -> None:
    """qb-compiler by QubitBoost — calibration-aware quantum circuit compiler."""


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
def compile(circuit: str, backend: str | None, output: str | None, strategy: str) -> None:
    """Compile a QASM circuit file."""
    from qb_compiler.compiler import QBCircuit, QBCompiler

    path = Path(circuit)
    content = path.read_text()

    # Simple QASM 2.0 parsing: count qubits from qreg declarations
    n_qubits = _parse_qasm_n_qubits(content)
    if n_qubits < 1:
        click.echo("Error: could not determine qubit count from QASM file.", err=True)
        sys.exit(1)

    qbc = QBCircuit(n_qubits)
    # Parse gates from QASM (lightweight parser)
    for gate_name, qubits, params in _parse_qasm_gates(content, n_qubits):
        qbc.add(gate_name, qubits, params)

    compiler = QBCompiler(backend=backend, strategy=strategy)
    result = compiler.compile(qbc)

    click.echo(f"Compiled: depth {result.original_depth} -> {result.compiled_depth} "
               f"({result.depth_reduction_pct:.1f}% reduction)")
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


# ── QASM parsing helpers ─────────────────────────────────────────────


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
    """Minimal QASM 2.0 gate extractor.

    Returns a list of (gate_name, qubit_tuple, param_tuple).
    This handles the most common patterns but is NOT a full parser.
    """
    import re

    gates: list[tuple[str, tuple[int, ...], tuple[float, ...]]] = []
    # Match lines like:  h q[0];  cx q[0],q[1];  rz(0.5) q[0];
    pattern = re.compile(
        r"^(\w+)"  # gate name
        r"(?:\(([^)]*)\))?"  # optional params in parens
        r"\s+"
        r"([\w\[\],\s]+)"  # qubit args
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
