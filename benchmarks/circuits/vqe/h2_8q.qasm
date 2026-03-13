OPENQASM 2.0;
include "qelib1.inc";

// 8-qubit VQE ansatz for H2 molecule (UCCSD-inspired)
// Hartree-Fock reference |11110000> with variational rotations

qreg q[8];
creg c[8];

// Hartree-Fock initial state: |11110000>
x q[0];
x q[1];
x q[2];
x q[3];

// Layer 1: single-excitation rotations
ry(0.3217) q[0];
ry(-0.1542) q[1];
ry(0.4891) q[2];
ry(-0.2763) q[3];
ry(0.1105) q[4];
ry(-0.0832) q[5];
ry(0.2219) q[6];
ry(-0.1478) q[7];

// Layer 2: entangling block (CNOT ladder)
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];

// Layer 3: second rotation layer
ry(0.1893) q[0];
ry(-0.2651) q[1];
ry(0.3104) q[2];
ry(-0.0917) q[3];
ry(0.4322) q[4];
ry(-0.1789) q[5];
ry(0.0654) q[6];
ry(-0.3541) q[7];

// Layer 4: reverse entangling block
cx q[7],q[6];
cx q[6],q[5];
cx q[5],q[4];
cx q[4],q[3];
cx q[3],q[2];
cx q[2],q[1];
cx q[1],q[0];

// Final Z-rotations
rz(0.7854) q[0];
rz(0.7854) q[1];
rz(0.7854) q[2];
rz(0.7854) q[3];
rz(0.7854) q[4];
rz(0.7854) q[5];
rz(0.7854) q[6];
rz(0.7854) q[7];

// Measurement
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];
measure q[6] -> c[6];
measure q[7] -> c[7];
