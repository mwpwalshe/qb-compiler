OPENQASM 2.0;
include "qelib1.inc";

// 10-qubit random circuit, approximate depth 30
// Mix of single-qubit (h, rz, ry, rx, x) and two-qubit (cx) gates
// Designed for stress-testing compilation passes

qreg q[10];
creg c[10];

// Depth 1-3
h q[0];
h q[2];
h q[4];
h q[6];
h q[8];
cx q[1],q[3];
cx q[5],q[7];
rz(1.2566) q[9];

// Depth 4-6
cx q[0],q[1];
cx q[4],q[5];
ry(0.8378) q[2];
rz(2.0944) q[3];
rx(1.5708) q[6];
cx q[8],q[9];
h q[7];

// Depth 7-9
cx q[2],q[3];
cx q[6],q[7];
rz(0.4189) q[0];
ry(1.6755) q[1];
h q[4];
rx(0.9425) q[5];
cx q[9],q[8];

// Depth 10-12
cx q[0],q[2];
cx q[4],q[6];
cx q[8],q[7];
ry(2.3562) q[1];
rz(0.7854) q[3];
rx(1.2566) q[5];
h q[9];

// Depth 13-15
cx q[1],q[0];
cx q[3],q[4];
cx q[5],q[6];
cx q[7],q[9];
rz(1.8850) q[2];
ry(0.5236) q[8];

// Depth 16-18
h q[0];
cx q[2],q[1];
cx q[6],q[8];
rz(2.5133) q[3];
ry(0.3491) q[4];
rx(1.0472) q[5];
cx q[9],q[7];

// Depth 19-21
cx q[0],q[3];
cx q[1],q[5];
cx q[7],q[8];
rz(0.6981) q[2];
h q[4];
ry(1.3090) q[6];
rx(2.7925) q[9];

// Depth 22-24
cx q[2],q[4];
cx q[6],q[9];
cx q[3],q[0];
rz(1.1519) q[1];
ry(0.2618) q[5];
h q[7];
rx(1.8326) q[8];

// Depth 25-27
cx q[0],q[1];
cx q[4],q[5];
cx q[7],q[6];
cx q[8],q[9];
rz(2.0071) q[2];
ry(0.9948) q[3];

// Depth 28-30
cx q[1],q[2];
cx q[3],q[5];
cx q[6],q[8];
cx q[9],q[0];
rz(0.5585) q[4];
h q[7];

// Final single-qubit layer
rz(1.4399) q[0];
ry(0.7330) q[1];
rx(2.1642) q[2];
h q[3];
rz(0.8901) q[4];
ry(1.7802) q[5];
rx(0.4014) q[6];
rz(2.6180) q[7];
ry(1.0996) q[8];
h q[9];

// Measurement
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];
measure q[6] -> c[6];
measure q[7] -> c[7];
measure q[8] -> c[8];
measure q[9] -> c[9];
