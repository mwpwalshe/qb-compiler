OPENQASM 2.0;
include "qelib1.inc";

// 5-qubit random circuit, approximate depth 20
// Mix of single-qubit (h, rz, ry, x) and two-qubit (cx) gates
// Seed-deterministic layout for reproducible benchmarking

qreg q[5];
creg c[5];

// Depth 1-2
h q[0];
h q[2];
h q[4];
cx q[1],q[3];

// Depth 3-4
rz(0.9273) q[0];
ry(1.2045) q[2];
cx q[3],q[4];
x q[1];

// Depth 5-6
cx q[0],q[1];
rz(2.1547) q[4];
ry(0.4812) q[3];
h q[2];

// Depth 7-8
cx q[2],q[3];
rz(1.7321) q[0];
ry(0.8917) q[1];
x q[4];

// Depth 9-10
cx q[4],q[0];
h q[3];
rz(0.3142) q[2];
ry(2.5133) q[1];

// Depth 11-12
cx q[1],q[2];
rz(1.0472) q[4];
h q[0];
ry(1.8850) q[3];

// Depth 13-14
cx q[3],q[4];
cx q[0],q[1];
rz(0.6283) q[2];

// Depth 15-16
ry(1.4137) q[0];
cx q[2],q[3];
rz(2.8274) q[1];
h q[4];

// Depth 17-18
cx q[4],q[0];
ry(0.7540) q[3];
rz(1.5708) q[2];
x q[1];

// Depth 19-20
cx q[1],q[3];
h q[0];
rz(0.2094) q[4];
ry(1.9635) q[2];

// Measurement
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
