function setup() {
  const canvas = createCanvas(850, 450);
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
  drawLossGraph();
  if (training) {
    updateFormulaPanel();
  }
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

  setTimeout(runEpoch, 1500); // delay for animation
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
    `Epoch ${epoch} / ${totalEpochs}`;
  document.getElementById("formula-text").innerHTML =
    `\\[ y = ${y.toFixed(2)}, \\quad \\hat{y} = ${yHat.toFixed(2)}, \\quad L = ${loss.toFixed(4)} \\]`;
  MathJax.typesetPromise();
}
