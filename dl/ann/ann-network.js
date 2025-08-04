// Fixed dataset: Celsius → Fahrenheit
let x = 50;
let y = 1.8 * x + 32;
let learningRate = 0.01;
let loss = 0;

let epoch = 0;
const totalEpochs = 4;

let lossHistory = [];

// --- Network Weights and Biases ---
// Input → Hidden1 (3 neurons)
let w1 = [Math.random(), Math.random(), Math.random()];
let b1 = [0, 0, 0];

// Hidden1 → Hidden2 (2 neurons)
let w2 = [
  [Math.random(), Math.random(), Math.random()],  // h2[0] connections
  [Math.random(), Math.random(), Math.random()]   // h2[1] connections
];
let b2 = [0, 0];

// Hidden2 → Output (1 neuron)
let w3 = [Math.random(), Math.random()];
let b3 = 0;

// --- Activations ---
let h1 = [0, 0, 0];
let h2 = [0, 0];
let yHat = 0;

// --- Activation Functions ---
function relu(z) {
  return Math.max(0, z);
}
function reluDerivative(z) {
  return z > 0 ? 1 : 0;
}

// --- Forward Pass ---
function forward(x) {
  // Input → Hidden1
  for (let i = 0; i < 3; i++) {
    h1[i] = relu(w1[i] * x + b1[i]);
  }

  // Hidden1 → Hidden2
  for (let j = 0; j < 2; j++) {
    let z = 0;
    for (let i = 0; i < 3; i++) {
      z += w2[j][i] * h1[i];
    }
    h2[j] = relu(z + b2[j]);
  }

  // Hidden2 → Output
  yHat = w3[0] * h2[0] + w3[1] * h2[1] + b3;
}

// --- Backpropagation ---
function backward(x, y) {
  forward(x);
  const error = yHat - y;

  // Gradients for output layer
  const dw3 = [error * h2[0], error * h2[1]];
  const db3 = error;

  // Gradients for hidden layer 2
  const dh2 = [
    w3[0] * error * reluDerivative(
      w2[0][0] * h1[0] + w2[0][1] * h1[1] + w2[0][2] * h1[2] + b2[0]
    ),
    w3[1] * error * reluDerivative(
      w2[1][0] * h1[0] + w2[1][1] * h1[1] + w2[1][2] * h1[2] + b2[1]
    )
  ];

  const dw2 = [
    [dh2[0] * h1[0], dh2[0] * h1[1], dh2[0] * h1[2]],
    [dh2[1] * h1[0], dh2[1] * h1[1], dh2[1] * h1[2]]
  ];
  const db2_new = dh2;

  // Gradients for hidden layer 1
  const dh1 = [0, 0, 0];
  for (let i = 0; i < 3; i++) {
    let dz = 0;
    for (let j = 0; j < 2; j++) {
      dz += dh2[j] * w2[j][i];
    }
    dh1[i] = dz * reluDerivative(w1[i] * x + b1[i]);
  }

  const dw1 = dh1.map(d => d * x);
  const db1_new = dh1;

  // --- Apply updates ---
  for (let i = 0; i < 2; i++) {
    w3[i] -= learningRate * dw3[i];
  }
  b3 -= learningRate * db3;

  for (let j = 0; j < 2; j++) {
    for (let i = 0; i < 3; i++) {
      w2[j][i] -= learningRate * dw2[j][i];
    }
    b2[j] -= learningRate * db2_new[j];
  }

  for (let i = 0; i < 3; i++) {
    w1[i] -= learningRate * dw1[i];
    b1[i] -= learningRate * db1_new[i];
  }

  // Loss tracking
  loss = (yHat - y) ** 2;
  lossHistory.push({ epoch, loss });
}
