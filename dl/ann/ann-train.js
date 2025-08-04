let x = 50;
let y = 1.8 * x + 32;
let learningRate = 0.01;
let loss = 0;

let epoch = 0, totalEpochs = 5;
let step = 0;
let training = false;

// Initialize network weights and biases
let w1 = [Math.random(), Math.random()];  // weights input → hidden
let b1 = [0, 0];                          // biases for hidden layer
let w2 = [Math.random(), Math.random()];  // weights hidden → output
let b2 = 0;                               // bias for output

// Activations (updated per step)
let h = [0, 0];   // hidden activations
let yHat = 0;     // predicted output

// Forward pass computation
function forward(x) {
  h = [
    relu(w1[0] * x + b1[0]),
    relu(w1[1] * x + b1[1])
  ];
  yHat = w2[0] * h[0] + w2[1] * h[1] + b2;
}

// ReLU activation
function relu(z) {
  return Math.max(0, z);
}

// Derivative of ReLU
function reluDerivative(z) {
  return z > 0 ? 1 : 0;
}

// Backward pass and update
function backward(x, y) {
  forward(x);  // Ensure latest forward pass
  const error = yHat - y;

  // Gradients for output weights and bias
  const dw2 = [error * h[0], error * h[1]];
  const db2 = error;

  // Gradients for hidden layer
  const dh = [
    error * w2[0] * reluDerivative(w1[0] * x + b1[0]),
    error * w2[1] * reluDerivative(w1[1] * x + b1[1])
  ];

  const dw1 = [dh[0] * x, dh[1] * x];
  const db1 = [dh[0], dh[1]];

  // Apply updates
  for (let i = 0; i < 2; i++) {
    w2[i] -= learningRate * dw2[i];
    w1[i] -= learningRate * dw1[i];
    b1[i] -= learningRate * db1[i];
  }
  b2 -= learningRate * db2;

  // Calculate new loss
  loss = (yHat - y) ** 2;
}

function setup() {
  const canvas = createCanvas(700, 400);
  canvas.parent("sketch-holder");
  textAlign(CENTER, CENTER);
  textSize(14);
  forward(x);
  updateDisplays();
}

function draw() {
  background(255);
  drawNetwork();

  if (training) updateFormulaPanel();
}

function drawNetwork() {
  const cx = 150, cy = 200;

  // Draw input
  drawNeuron(cx - 120, cy, "x", x);

  // Draw hidden neurons
  for (let i = 0; i < 2; i++) {
    const hx = cx, hy = cy - 60 + i * 120;
    drawNeuron(hx, hy, "h" + (i + 1), h[i]);
    drawArrow(cx - 90, cy, hx - 30, hy, true, null, map(Math.abs(w1[i]), 0, 5, 1, 6));
  }

  // Draw output neuron
  drawNeuron(cx + 120, cy, "ŷ", yHat);
  for (let i = 0; i < 2; i++) {
    const hx = cx, hy = cy - 60 + i * 120;
    drawArrow(hx + 30, hy, cx + 90, cy, true, null, map(Math.abs(w2[i]), 0, 5, 1, 6));
  }
}

function startTraining() {
  epoch = 0;
  training = true;
  runEpoch();
}

function runEpoch() {
  if (epoch >= totalEpochs) {
    training = false;
    return;
  }

  forward(x);
  backward(x, y);
  updateDisplays();
  epoch++;

  setTimeout(runEpoch, 1000); // slower for visibility
}


function updateFormulaPanel() {
  document.getElementById("step-text").innerText = `Epoch ${epoch} / ${totalEpochs}`;
  document.getElementById("formula-text").innerHTML = `\\[ y = ${y.toFixed(2)}, \\quad \\hat{y} = ${yHat.toFixed(2)}, \\quad L = ${loss.toFixed(4)} \\]`;
  MathJax.typesetPromise();
}

function updateDisplays() {
  document.getElementById("weightDisplay").innerText = `w1: [${w1.map(w => w.toFixed(2)).join(", ")}], w2: [${w2.map(w => w.toFixed(2)).join(", ")}]`;
  document.getElementById("biasDisplay").innerText = `b1: [${b1.map(b => b.toFixed(2)).join(", ")}], b2: ${b2.toFixed(2)}`;
  document.getElementById("lossDisplay").innerText = `Loss: ${loss.toFixed(4)}`;
}

function updateInput() {
  x = parseInt(document.getElementById("inputC").value);
  document.getElementById("inputCVal").innerText = x;
  y = 1.8 * x + 32;
  forward(x);
}
