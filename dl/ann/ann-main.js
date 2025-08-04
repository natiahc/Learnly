let training = false;
let stepIndex = 0;
let stepDelay = 800; // ms between steps
let currentPhase = "forward"; // forward, loss, backward
let activeConnections = []; // store which arrows to highlight

function setup() {
  const canvas = createCanvas(950, 600);
  canvas.parent("sketch-holder");
  textAlign(CENTER, CENTER);
  textSize(14);
  forward(x); // initial forward pass
  updateDisplays();
  startTraining();
}

function draw() {
  background(255);
  drawNetwork();
  drawLossGraph();
  updateFormulaPanel();
}

function startTraining() {
  training = true;
  epoch = 0;
  stepIndex = 0;
  currentPhase = "forward";
  activeConnections = [];
  runStep();
}

function runStep() {
  if (!training) return;

  if (currentPhase === "forward") {
    if (stepIndex === 0) {
      // Input → Hidden1
      activeConnections = [];
      for (let i = 0; i < 3; i++) {
        activeConnections.push({ from: "x", to: `h1${i+1}` });
      }
    } 
    else if (stepIndex === 1) {
      // Hidden1 → Hidden2
      activeConnections = [];
      for (let j = 0; j < 2; j++) {
        for (let i = 0; i < 3; i++) {
          activeConnections.push({ from: `h1${i+1}`, to: `h2${j+1}` });
        }
      }
    } 
    else if (stepIndex === 2) {
      // Hidden2 → Output
      activeConnections = [];
      for (let j = 0; j < 2; j++) {
        activeConnections.push({ from: `h2${j+1}`, to: "yhat" });
      }
    } 
    else if (stepIndex === 3) {
      // Calculate loss
      forward(x);
      loss = (yHat - y) ** 2;
      activeConnections = [];
      currentPhase = "backward";
      stepIndex = -1; // reset for backward steps
    }
  } 
  else if (currentPhase === "backward") {
    if (stepIndex === 0) {
      // Output → Hidden2
      activeConnections = [];
      for (let j = 0; j < 2; j++) {
        activeConnections.push({ from: "yhat", to: `h2${j+1}`, color: "red" });
      }
    } 
    else if (stepIndex === 1) {
      // Hidden2 → Hidden1
      activeConnections = [];
      for (let j = 0; j < 2; j++) {
        for (let i = 0; i < 3; i++) {
          activeConnections.push({ from: `h2${j+1}`, to: `h1${i+1}`, color: "red" });
        }
      }
    } 
    else if (stepIndex === 2) {
      // Hidden1 → Input
      activeConnections = [];
      for (let i = 0; i < 3; i++) {
        activeConnections.push({ from: `h1${i+1}`, to: "x", color: "red" });
      }
    } 
    else if (stepIndex === 3) {
      // Apply weight updates
      backward(x, y);
      updateDisplays();
      epoch++;
      if (epoch >= totalEpochs) {
        training = false;
        return;
      }
      currentPhase = "forward";
      stepIndex = -1;
    }
  }

  stepIndex++;
  setTimeout(runStep, stepDelay);
}

function updateDisplays() {
  document.getElementById("weightDisplay").innerText =
    `w1: [${w1.map(v => v.toFixed(2)).join(", ")}], ` +
    `w2: [${w2.map(row => "[" + row.map(v => v.toFixed(2)).join(", ") + "]").join(", ")}], ` +
    `w3: [${w3.map(v => v.toFixed(2)).join(", ")}]`;

  document.getElementById("biasDisplay").innerText =
    `b1: [${b1.map(v => v.toFixed(2)).join(", ")}], ` +
    `b2: [${b2.map(v => v.toFixed(2)).join(", ")}], b3: ${b3.toFixed(2)}`;

  document.getElementById("lossDisplay").innerText =
    `Loss: ${loss.toFixed(4)}`;
}

function updateFormulaPanel() {
  document.getElementById("step-text").innerText =
    `Epoch ${epoch+1} / ${totalEpochs} | Phase: ${currentPhase}`;
  document.getElementById("formula-text").innerHTML =
    `\\[ y = ${y.toFixed(2)}, \\quad \\hat{y} = ${yHat.toFixed(2)}, \\quad L = ${loss.toFixed(4)} \\]`;
  MathJax.typesetPromise();
}
