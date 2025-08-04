let x = 50;
let y = 1.8 * x + 32;
let learningRate = 0.01;
let loss = 0;

let epoch = 0;
const totalEpochs = 4;

let w1 = [Math.random(), Math.random(), Math.random()];
let b1 = [0, 0, 0];
let w2 = [Math.random(), Math.random(), Math.random()];
let b2 = 0;

let h = [0, 0, 0];
let yHat = 0;
let training = false;

function relu(z) {
  return Math.max(0, z);
}
function reluDerivative(z) {
  return z > 0 ? 1 : 0;
}

function forward(x) {
  for (let i = 0; i < 3; i++) {
    h[i] = relu(w1[i] * x + b1[i]);
  }
  yHat = w2[0] * h[0] + w2[1] * h[1] + w2[2] * h[2] + b2;
}

function backward(x, y) {
  forward(x);
  const error = yHat - y;

  const dw2 = h.map(hj => error * hj);
  const db2 = error;

  const dh = [];
  for (let i = 0; i < 3; i++) {
    const dz = w2[i] * error * reluDerivative(w1[i] * x + b1[i]);
    dh.push(dz);
  }

  const dw1 = dh.map(d => d * x);
  const db1_new = dh;

  for (let i = 0; i < 3; i++) {
    w2[i] -= learningRate * dw2[i];
    w1[i] -= learningRate * dw1[i];
    b1[i] -= learningRate * db1_new[i];
  }
  b2 -= learningRate * db2;

  loss = (yHat - y) ** 2;
}

// -------------------- p5.js part --------------------

function setup() {
  const canvas = createCanvas(800, 400);
  canvas.parent("sketch-holder");
  textAlign(CENTER, CENTER);
  textSize(14);
  forward(x);
  updateDisplays();
  training = true;
  runEpoch();
}

function draw() {
  background(255);
  drawNetwork();
  if (training) updateFormulaPanel();
}

function drawNetwork() {
  const cx = 200, cy = 200;
  drawNeuron(cx - 120, cy, "x", x);

  for (let i = 0; i < 3; i++) {
    const hy = cy - 80 + i * 80;
    drawNeuron(cx, hy, `h${i + 1}`, h[i]);
    drawArrow(cx - 90, cy, cx - 30, hy, true, null, map(Math.abs(w1[i]), 0, 5, 1, 6));
  }

  drawNeuron(cx + 160, cy, "ŷ", yHat);
  for (let i = 0; i < 3; i++) {
    const hy = cy - 80 + i * 80;
    drawArrow(cx + 30, hy, cx + 130, cy, true, null, map(Math.abs(w2[i]), 0, 5, 1, 6));
  }
}

function drawNeuron(x, y, label, value = null) {
  stroke(0);
  fill("#eef");
  ellipse(x, y, 60);
  fill(0);
  noStroke();
  text(label, x, y - 10);
  if (value !== null) {
    text(`=${value.toFixed(2)}`, x, y + 12);
  }
}

function drawArrow(x1, y1, x2, y2, active = true, colorOverride = null, thickness = 1) {
  stroke(colorOverride || (active ? "#0077cc" : "#aaa"));
  strokeWeight(thickness);
  line(x1, y1, x2, y2);
  const angle = atan2(y2 - y1, x2 - x1);
  push();
  translate(x2, y2);
  rotate(angle);
  fill(colorOverride || (active ? "#0077cc" : "#aaa"));
  triangle(0, 0, -10, 5, -10, -5);
  pop();
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

  setTimeout(runEpoch, 1500);
}

function updateDisplays() {
  document.getElementById("weightDisplay").innerText = `w1: [${w1.map(w => w.toFixed(2)).join(", ")}], w2: [${w2.map(w => w.toFixed(2)).join(", ")}]`;
  document.getElementById("biasDisplay").innerText = `b1: [${b1.map(b => b.toFixed(2)).join(", ")}], b2: ${b2.toFixed(2)}`;
  document.getElementById("lossDisplay").innerText = `Loss: ${loss.toFixed(4)}`;
}

function updateFormulaPanel() {
  document.getElementById("step-text").innerText = `Epoch ${epoch} / ${totalEpochs}`;
  document.getElementById("formula-text").innerHTML = `\\[ y = ${y.toFixed(2)}, \\quad \\hat{y} = ${yHat.toFixed(2)}, \\quad L = ${loss.toFixed(4)} \\]`;
  MathJax.typesetPromise();
}
