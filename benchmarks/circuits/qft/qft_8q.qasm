OPENQASM 2.0;
include "qelib1.inc";

// 8-qubit Quantum Fourier Transform
// Standard QFT decomposition: H + controlled-phase gates + final swaps

qreg q[8];
creg c[8];

// Prepare input state
x q[0];
x q[3];
x q[5];

// QFT on qubits 0..7
// Qubit 0
h q[0];
cp(1.5707963) q[1],q[0];
cp(0.7853982) q[2],q[0];
cp(0.3926991) q[3],q[0];
cp(0.1963495) q[4],q[0];
cp(0.0981748) q[5],q[0];
cp(0.0490874) q[6],q[0];
cp(0.0245437) q[7],q[0];

// Qubit 1
h q[1];
cp(1.5707963) q[2],q[1];
cp(0.7853982) q[3],q[1];
cp(0.3926991) q[4],q[1];
cp(0.1963495) q[5],q[1];
cp(0.0981748) q[6],q[1];
cp(0.0490874) q[7],q[1];

// Qubit 2
h q[2];
cp(1.5707963) q[3],q[2];
cp(0.7853982) q[4],q[2];
cp(0.3926991) q[5],q[2];
cp(0.1963495) q[6],q[2];
cp(0.0981748) q[7],q[2];

// Qubit 3
h q[3];
cp(1.5707963) q[4],q[3];
cp(0.7853982) q[5],q[3];
cp(0.3926991) q[6],q[3];
cp(0.1963495) q[7],q[3];

// Qubit 4
h q[4];
cp(1.5707963) q[5],q[4];
cp(0.7853982) q[6],q[4];
cp(0.3926991) q[7],q[4];

// Qubit 5
h q[5];
cp(1.5707963) q[6],q[5];
cp(0.7853982) q[7],q[5];

// Qubit 6
h q[6];
cp(1.5707963) q[7],q[6];

// Qubit 7
h q[7];

// Swap to reverse qubit order (standard QFT convention)
swap q[0],q[7];
swap q[1],q[6];
swap q[2],q[5];
swap q[3],q[4];

// Measurement
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];
measure q[6] -> c[6];
measure q[7] -> c[7];
