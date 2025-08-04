let x = 50;
let y = 1.8 * x + 32;
let w = 0.1, b = 0.0;
let learningRate = 0.01;
let loss = 0;

let epoch = 0, totalEpochs = 5;
let step = 0;
let training = false;

const steps = [
  () => {}, // Step 0: input
  () => {}, // Step 1: target
  () => {}, // Step 2: prediction
  () => {}, // Step 3: loss
  () => {}, // Step 4: gradient
  () => {   // Step 5: update weight
    const dw = 2 * (w * x + b - y) * x;
    w -= learningRate * dw;
  },
  () => {   // Step 6: update bias
    const db = 2 * (w * x + b - y);
    b -= learningRate * db;
  }
];

const formulas = [
  () => `x = ${x}`,
  () => `y = ${y.toFixed(2)}`,
  () => `\\hat{y} = wx + b = ${w.toFixed(2)}·${x} + ${b.toFixed(2)} = ${(w * x + b).toFixed(2)}`,
  () => {
    const pred = w * x + b;
    loss = (pred - y) ** 2;
    return `L = (ŷ - y)^2 = (${pred.toFixed(2)} - ${y.toFixed(2)})^2 = ${loss.toFixed(4)}`;
  },
  () => {
    const dL = 2 * (w * x + b - y);
    return `\\frac{\\partial L}{\\partial w} = 2(ŷ - y)·x = ${dL.toFixed(2)} × ${x} = ${(dL * x).toFixed(2)}`;
  },
  () => {
    const dw = 2 * (w * x + b - y) * x;
    return `w = w - η·∂L/∂w = ${w.toFixed(2)} - ${learningRate} × ${dw.toFixed(2)}`;
  },
  () => {
    const db = 2 * (w * x + b - y);
    return `b = b - η·∂L/∂b = ${b.toFixed(2)} - ${learningRate} × ${db.toFixed(2)}`;
  }
];

function setup() {
  const canvas = createCanvas(700, 300);
  canvas.parent("sketch-holder");
  textAlign(CENTER, CENTER);
  textSize(14);
  updateDisplays();
}

function draw() {
  background(255);
  drawNetwork();

  if (training) updateFormulaPanel();
}

function drawNetwork() {
  // Neuron: input and output
  drawNeuron(100, 150, "x", x);
  const yHat = w * x + b;
  drawNeuron(300, 150, "ŷ", yHat);

  // Arrow with thickness based on weight
  const weightThickness = map(Math.abs(w), 0, 5, 1, 8);
  drawArrow(130, 150, 270, 150, step >= 2, null, weightThickness);

  // Backward arrow if step >= 5
  if (step >= 5) {
    drawArrow(270, 150, 130, 150, true, "red", 2);
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

function updateFormulaPanel() {
  document.getElementById("step-text").innerText = `Epoch ${epoch + 1}, Step ${step + 1} of 7`;
  document.getElementById("formula-text").innerHTML = `\\[ ${formulas[step]()} \\]`;
  MathJax.typesetPromise();
  updateDisplays();
}

function updateDisplays() {
  document.getElementById("weightDisplay").innerText = `Weight: ${w.toFixed(4)}`;
  document.getElementById("biasDisplay").innerText = `Bias: ${b.toFixed(4)}`;
  document.getElementById("lossDisplay").innerText = `Loss: ${loss.toFixed(4)}`;
}

function updateInput() {
  x = parseInt(document.getElementById("inputC").value);
  document.getElementById("inputCVal").innerText = x;
  y = 1.8 * x + 32;
}

function updateEpochs() {
  totalEpochs = parseInt(document.getElementById("epochSelect").value);
}

function startTraining() {
  epoch = 0;
  step = 0;
  training = true;
  runEpoch();
}

function runEpoch() {
  if (epoch >= totalEpochs) {
    training = false;
    return;
  }

  step = 0;

  function nextStep() {
    if (step >= steps.length) {
      epoch++;
      runEpoch();
      return;
    }

    steps[step]();  // execute logic for this step
    updateFormulaPanel();
    step++;
    setTimeout(nextStep, 1500);
  }

  nextStep();
}
