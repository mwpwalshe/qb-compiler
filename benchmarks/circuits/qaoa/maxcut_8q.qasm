OPENQASM 2.0;
include "qelib1.inc";

// 8-qubit QAOA MaxCut on an 8-node ring graph (0-1-2-3-4-5-6-7-0)
// p=2 layers with gamma1=0.8, beta1=0.5, gamma2=0.6, beta2=0.4

qreg q[8];
creg c[8];

// Initial superposition
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];

// ---- Layer 1 (gamma1=0.8) ----
// Problem unitary: ZZ interactions for ring edges
// Edge (0,1)
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

// Edge (3,4)
cx q[3],q[4];
rz(1.6) q[4];
cx q[3],q[4];

// Edge (4,5)
cx q[4],q[5];
rz(1.6) q[5];
cx q[4],q[5];

// Edge (5,6)
cx q[5],q[6];
rz(1.6) q[6];
cx q[5],q[6];

// Edge (6,7)
cx q[6],q[7];
rz(1.6) q[7];
cx q[6],q[7];

// Edge (7,0)
cx q[7],q[0];
rz(1.6) q[0];
cx q[7],q[0];

// Mixer unitary (beta1=0.5)
rx(1.0) q[0];
rx(1.0) q[1];
rx(1.0) q[2];
rx(1.0) q[3];
rx(1.0) q[4];
rx(1.0) q[5];
rx(1.0) q[6];
rx(1.0) q[7];

// ---- Layer 2 (gamma2=0.6) ----
// Edge (0,1)
cx q[0],q[1];
rz(1.2) q[1];
cx q[0],q[1];

// Edge (1,2)
cx q[1],q[2];
rz(1.2) q[2];
cx q[1],q[2];

// Edge (2,3)
cx q[2],q[3];
rz(1.2) q[3];
cx q[2],q[3];

// Edge (3,4)
cx q[3],q[4];
rz(1.2) q[4];
cx q[3],q[4];

// Edge (4,5)
cx q[4],q[5];
rz(1.2) q[5];
cx q[4],q[5];

// Edge (5,6)
cx q[5],q[6];
rz(1.2) q[6];
cx q[5],q[6];

// Edge (6,7)
cx q[6],q[7];
rz(1.2) q[7];
cx q[6],q[7];

// Edge (7,0)
cx q[7],q[0];
rz(1.2) q[0];
cx q[7],q[0];

// Mixer unitary (beta2=0.4)
rx(0.8) q[0];
rx(0.8) q[1];
rx(0.8) q[2];
rx(0.8) q[3];
rx(0.8) q[4];
rx(0.8) q[5];
rx(0.8) q[6];
rx(0.8) q[7];

// Measurement
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];
measure q[6] -> c[6];
measure q[7] -> c[7];
