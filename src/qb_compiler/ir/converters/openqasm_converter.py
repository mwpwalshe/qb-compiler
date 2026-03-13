"""Convert between OpenQASM 2.0 strings and :class:`QBCircuit`.

This is a self-contained parser/emitter for the subset of QASM 2.0 that
matters in practice (gate applications, measurements, barriers, classical
conditions, and the standard header).  No external QASM library is needed.
"""

from __future__ import annotations

import math
import re
from typing import Any

from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.operations import QBGate

# ── QASM 2.0 parser ──────────────────────────────────────────────────

_HEADER_RE = re.compile(r"^OPENQASM\s+2\.0\s*;")
_INCLUDE_RE = re.compile(r'^include\s+"[^"]+"\s*;')
_QREG_RE = re.compile(r"^qreg\s+(\w+)\s*\[\s*(\d+)\s*\]\s*;")
_CREG_RE = re.compile(r"^creg\s+(\w+)\s*\[\s*(\d+)\s*\]\s*;")
_MEASURE_RE = re.compile(
    r"^measure\s+(\w+)\s*\[\s*(\d+)\s*\]\s*->\s*(\w+)\s*\[\s*(\d+)\s*\]\s*;"
)
_BARRIER_RE = re.compile(r"^barrier\s+(.*?)\s*;")
_IF_RE = re.compile(r"^if\s*\(\s*(\w+)\s*==\s*(\d+)\s*\)\s+(.*)")
_GATE_RE = re.compile(
    r"^(\w+)"                          # gate name
    r"(?:\s*\(([^)]*)\))?"             # optional params
    r"\s+([\w\[\],\s]+)\s*;"           # qubit args
)
_ARG_RE = re.compile(r"(\w+)\s*\[\s*(\d+)\s*\]")


def _parse_params(param_str: str) -> tuple[float, ...]:
    """Evaluate simple QASM parameter expressions (pi, *, /, +, -)."""
    if not param_str or not param_str.strip():
        return ()
    params: list[float] = []
    for token in param_str.split(","):
        token = token.strip()
        # Replace pi with its value, then eval safely
        expr = token.replace("pi", str(math.pi))
        try:
            val = float(_safe_eval(expr))
        except Exception:
            val = 0.0
        params.append(val)
    return tuple(params)


def _safe_eval(expr: str) -> float:
    """Evaluate a simple arithmetic expression with +, -, *, / and floats.

    Uses a tokenizer + recursive descent parser instead of ``eval()``
    to avoid code injection risks from user-supplied QASM files.
    """
    import ast
    import operator

    cleaned = expr.strip()
    # Fast path: if it's just a number, parse directly
    try:
        return float(cleaned)
    except ValueError:
        pass

    # Only allow digits, decimal points, +, -, *, /, whitespace, e (scientific)
    if not re.match(r"^[\d\.\+\-\*/eE\s]+$", cleaned):
        raise ValueError(f"Unsafe expression: {expr!r}")

    # Parse as an AST and evaluate only safe numeric operations
    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Cannot parse expression: {expr!r}") from exc

    _bin_ops: dict[type, Any] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }
    _unary_ops: dict[type, Any] = {
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def _eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _bin_ops:
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            return float(_bin_ops[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _unary_ops:
            return float(_unary_ops[type(node.op)](_eval_node(node.operand)))
        raise ValueError(f"Unsupported expression node: {ast.dump(node)}")

    return _eval_node(tree)


def _parse_args(
    arg_str: str,
    qubit_offsets: dict[str, int],
) -> tuple[int, ...]:
    """Parse ``q[0],q[1]`` style arguments into global qubit indices."""
    indices: list[int] = []
    for m in _ARG_RE.finditer(arg_str):
        reg_name = m.group(1)
        idx = int(m.group(2))
        offset = qubit_offsets.get(reg_name)
        if offset is None:
            raise ValueError(f"Unknown register '{reg_name}' in args: {arg_str}")
        indices.append(offset + idx)
    return tuple(indices)


def from_qasm(qasm_str: str) -> QBCircuit:
    """Parse an OpenQASM 2.0 string into a :class:`QBCircuit`.

    Supports: ``qreg``, ``creg``, standard gates, ``measure``, ``barrier``,
    ``if(c==val)`` conditions.  Gate definitions (``gate ... { }``) are
    skipped — only applications matter for the IR.
    """
    lines = _strip_comments(qasm_str)

    qubit_regs: dict[str, int] = {}   # reg_name -> size
    clbit_regs: dict[str, int] = {}
    qubit_offsets: dict[str, int] = {}  # reg_name -> global offset
    clbit_offsets: dict[str, int] = {}
    total_qubits = 0
    total_clbits = 0

    # First pass: collect registers
    for line in lines:
        m = _QREG_RE.match(line)
        if m:
            name, size = m.group(1), int(m.group(2))
            qubit_regs[name] = size
            qubit_offsets[name] = total_qubits
            total_qubits += size
            continue
        m = _CREG_RE.match(line)
        if m:
            name, size = m.group(1), int(m.group(2))
            clbit_regs[name] = size
            clbit_offsets[name] = total_clbits
            total_clbits += size

    if total_qubits == 0:
        raise ValueError("No qreg declarations found in QASM")

    circ = QBCircuit(n_qubits=total_qubits, n_clbits=max(total_clbits, 0))

    # Second pass: operations
    in_gate_def = False
    brace_depth = 0

    for line in lines:
        # Skip headers
        if _HEADER_RE.match(line) or _INCLUDE_RE.match(line):
            continue
        if _QREG_RE.match(line) or _CREG_RE.match(line):
            continue

        # Skip gate definitions
        if line.startswith("gate "):
            in_gate_def = True
            brace_depth += line.count("{") - line.count("}")
            continue
        if in_gate_def:
            brace_depth += line.count("{") - line.count("}")
            if brace_depth <= 0:
                in_gate_def = False
            continue

        # Condition prefix
        condition = None
        m_if = _IF_RE.match(line)
        if m_if:
            creg_name = m_if.group(1)
            cond_val = int(m_if.group(2))
            cond_offset = clbit_offsets.get(creg_name, 0)
            condition = (cond_offset, float(cond_val))
            line = m_if.group(3).strip()

        # Measurement
        m = _MEASURE_RE.match(line)
        if m:
            q_reg, q_idx = m.group(1), int(m.group(2))
            c_reg, c_idx = m.group(3), int(m.group(4))
            qubit = qubit_offsets.get(q_reg, 0) + q_idx
            clbit = clbit_offsets.get(c_reg, 0) + c_idx
            circ.add_measurement(qubit, clbit)
            continue

        # Barrier
        m = _BARRIER_RE.match(line)
        if m:
            qubits = _parse_args(m.group(1), qubit_offsets)
            circ.add_barrier(qubits)
            continue

        # Gate application
        m = _GATE_RE.match(line)
        if m:
            gate_name = m.group(1).lower()
            params = _parse_params(m.group(2) or "")
            qubits = _parse_args(m.group(3), qubit_offsets)
            if not qubits:
                continue
            gate = QBGate(
                name=gate_name,
                qubits=qubits,
                params=params,
                condition=condition,
            )
            circ.add_gate(gate)

    return circ


# ── QASM 2.0 emitter ─────────────────────────────────────────────────


def to_qasm(circuit: QBCircuit) -> str:
    """Emit an OpenQASM 2.0 string from a :class:`QBCircuit`.

    A single ``qreg q[N]`` and ``creg c[M]`` are used.
    """
    lines: list[str] = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{circuit.n_qubits}];",
    ]
    if circuit.n_clbits > 0:
        lines.append(f"creg c[{circuit.n_clbits}];")

    from qb_compiler.ir.operations import QBBarrier, QBMeasure

    for op in circuit.iter_ops():
        if isinstance(op, QBGate):
            line = _gate_to_qasm(op)
            lines.append(line)
        elif isinstance(op, QBMeasure):
            lines.append(f"measure q[{op.qubit}] -> c[{op.clbit}];")
        elif isinstance(op, QBBarrier):
            qubit_args = ",".join(f"q[{q}]" for q in op.qubits)
            lines.append(f"barrier {qubit_args};")

    lines.append("")  # trailing newline
    return "\n".join(lines)


def _gate_to_qasm(gate: QBGate) -> str:
    """Format a single gate as a QASM 2.0 line."""
    prefix = ""
    if gate.condition is not None:
        _clbit, val = gate.condition
        prefix = f"if(c=={int(val)}) "

    qubit_args = ",".join(f"q[{q}]" for q in gate.qubits)

    if gate.params:
        param_strs = []
        for p in gate.params:
            # Try to express in terms of pi for readability
            ratio = p / math.pi if math.pi != 0 else 0
            if abs(ratio - round(ratio)) < 1e-10 and abs(round(ratio)) <= 8:
                r = round(ratio)
                if r == 0:
                    param_strs.append("0")
                elif r == 1:
                    param_strs.append("pi")
                elif r == -1:
                    param_strs.append("-pi")
                else:
                    param_strs.append(f"{r}*pi")
            else:
                param_strs.append(f"{p:.15g}")
        param_part = f"({','.join(param_strs)})"
    else:
        param_part = ""

    return f"{prefix}{gate.name}{param_part} {qubit_args};"


# ── helpers ───────────────────────────────────────────────────────────


def _strip_comments(qasm: str) -> list[str]:
    """Remove // comments and blank lines, return trimmed lines."""
    result: list[str] = []
    for raw in qasm.splitlines():
        line = raw.split("//")[0].strip()
        if line:
            result.append(line)
    return result
