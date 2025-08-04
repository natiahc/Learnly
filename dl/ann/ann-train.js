let x = 50;               // Celsius input
let y = 1.8 * x + 32;     // Ground truth Fahrenheit output
let w = 0.1, b = 0.0;     // Initial weight and bias
let learningRate = 0.01;

let epoch = 0, totalEpochs = 5;
let step = 0;
let training = false;

let loss = 0;

const steps = [
  () => {}, // Step 0: x
  () => {}, // Step 1: y
  () => {}, // Step 2: prediction
  () => {}, // Step 3: loss
  () => {}, // Step 4: gradient
  () => {   // Step 5: update w
    let dw = 2 * (w * x + b - y) * x;
    w -= learningRate * dw;
  },
  () => {   // Step 6: update b
    let db = 2 * (w * x + b - y);
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
    return `\\frac{\\partial L}{\\partial w} = 2(ŷ - y)x = ${dL.toFixed(2)} × ${x} = ${(dL * x).toFixed(2)}`;
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
  drawNeuron(100, 150, `x = ${x}`);
  drawNeuron(300, 150, `ŷ`);
  drawArrow(130, 150, 270, 150, step >= 2);

  if (step >= 5) {
    drawArrow(270, 150, 130, 150, true, "red");
  }
}

function drawNeuron(x, y, label) {
  stroke(0);
  fill("#eef");
  ellipse(x, y, 60);
  fill(0);
  noStroke();
  text(label, x, y);
}

function drawArrow(x1, y1, x2, y2, active = true, colorOverride = null) {
  stroke(colorOverride || (active ? "#0077cc" : "#aaa"));
  strokeWeight(active ? 3 : 1);
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

    steps[step]();      // Execute logic for the step
    updateFormulaPanel();
    step++;
    setTimeout(nextStep, 1500);
  }

  nextStep();
}
