OPENQASM 2.0;
include "qelib1.inc";

// 4-qubit VQE ansatz for H2 molecule (minimal UCCSD-inspired)
// Hartree-Fock reference |1100> with single-excitation rotations

qreg q[4];
creg c[4];

// Hartree-Fock initial state: |1100>
x q[0];
x q[1];

// Layer 1: single-excitation rotations
ry(0.4215) q[0];
ry(-0.2103) q[1];
ry(0.3318) q[2];
ry(-0.1547) q[3];

// Entangling block: CNOT ladder
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];

// Layer 2: second rotation layer
ry(0.1892) q[0];
ry(-0.3051) q[1];
ry(0.2764) q[2];
ry(-0.0918) q[3];

// Reverse entangling block
cx q[3],q[2];
cx q[2],q[1];
cx q[1],q[0];

// Final Z-rotations
rz(0.7854) q[0];
rz(0.7854) q[1];
rz(0.7854) q[2];
rz(0.7854) q[3];

// Measurement
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
