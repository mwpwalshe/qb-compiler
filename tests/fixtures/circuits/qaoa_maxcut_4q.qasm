OPENQASM 2.0;
include "qelib1.inc";

// 4-qubit QAOA for MaxCut on a 4-node ring graph
// Single layer (p=1) with fixed angles gamma=0.8, beta=0.5

qreg q[4];
creg c[4];

// Initial superposition
h q[0];
h q[1];
h q[2];
h q[3];

// Problem unitary: ZZ interactions for edges (0,1), (1,2), (2,3), (3,0)
// Edge (0,1): e^{-i * gamma * Z_0 Z_1}
cx q[0],q[1];
rz(1.6) q[1];
cx q[0],q[1];

// Edge (1,2)
cx q[1],q[2];
rz(1.6) q[2];
cx q[1],q[2];

// Edge (2,3)
cx q[2],q[3];
rz(1.6) q[3];
cx q[2],q[3];

// Edge (3,0)
cx q[3],q[0];
rz(1.6) q[0];
cx q[3],q[0];

// Mixer unitary: e^{-i * beta * X_j} for each qubit
rx(1.0) q[0];
rx(1.0) q[1];
rx(1.0) q[2];
rx(1.0) q[3];

// Measurement
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
