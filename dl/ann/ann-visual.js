function drawNetwork() {
  const layerX = [150, 350, 550, 750]; // x positions for each layer
  const layerY_hidden1 = [140, 220, 300]; // y positions for hidden1 neurons
  const layerY_hidden2 = [180, 260];      // y positions for hidden2 neurons
  const centerY = 220;                    // output neuron y position

  // Input neuron
  drawNeuron(layerX[0], centerY, "x", x);

  // Hidden1 neurons + connections from input
  for (let i = 0; i < 3; i++) {
    drawNeuron(layerX[1], layerY_hidden1[i], `h1${i + 1}`, h1[i]);
    drawArrow(
      layerX[0] + 30, centerY,
      layerX[1] - 30, layerY_hidden1[i],
      true,
      null,
      map(Math.abs(w1[i]), 0, 5, 1, 6)
    );
  }

  // Hidden2 neurons + connections from hidden1
  for (let j = 0; j < 2; j++) {
    drawNeuron(layerX[2], layerY_hidden2[j], `h2${j + 1}`, h2[j]);
    for (let i = 0; i < 3; i++) {
      drawArrow(
        layerX[1] + 30, layerY_hidden1[i],
        layerX[2] - 30, layerY_hidden2[j],
        true,
        null,
        map(Math.abs(w2[j][i]), 0, 5, 1, 6)
      );
    }
  }

  // Output neuron + connections from hidden2
  drawNeuron(layerX[3], centerY, "ŷ", yHat);
  for (let j = 0; j < 2; j++) {
    drawArrow(
      layerX[2] + 30, layerY_hidden2[j],
      layerX[3] - 30, centerY,
      true,
      null,
      map(Math.abs(w3[j]), 0, 5, 1, 6)
    );
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

function drawLossGraph() {
  const graphX = 50, graphY = 350, graphW = 700, graphH = 70;
  const maxLoss = Math.max(...lossHistory.map(p => p.loss), 1);

  // Axis
  stroke(0);
  noFill();
  rect(graphX, graphY, graphW, graphH);

  // Labels
  noStroke();
  fill(0);
  text("Loss", graphX - 20, graphY + graphH / 2);
  text("Epoch", graphX + graphW / 2, graphY + graphH + 15);

  // Plot line
  stroke("#cc0000");
  noFill();
  beginShape();
  for (let i = 0; i < lossHistory.length; i++) {
    const px = graphX + (i / (totalEpochs - 1)) * graphW;
    const py = graphY + graphH - (lossHistory[i].loss / maxLoss) * graphH;
    vertex(px, py);
  }
  endShape();
}
