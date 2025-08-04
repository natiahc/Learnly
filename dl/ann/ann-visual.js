function drawNetwork() {
  const layerX = [150, 400, 650, 900];
  const layerY_hidden1 = [140, 260, 380];
  const layerY_hidden2 = [200, 320];
  const centerY = 260;

  // Draw neurons
  drawNeuron(layerX[0], centerY, "x", x);
  for (let i = 0; i < 3; i++) drawNeuron(layerX[1], layerY_hidden1[i], `h1${i+1}`, h1[i]);
  for (let j = 0; j < 2; j++) drawNeuron(layerX[2], layerY_hidden2[j], `h2${j+1}`, h2[j]);
  drawNeuron(layerX[3], centerY, "ŷ", yHat);

  // Draw all connections (gray by default)
  drawAllConnections(layerX, layerY_hidden1, layerY_hidden2, centerY);

  // Highlight active connections
  if (Array.isArray(activeConnections)) {
    for (let conn of activeConnections) {
      highlightConnection(conn, layerX, layerY_hidden1, layerY_hidden2, centerY);
    }
  }
}

function drawAllConnections(layerX, layerY_hidden1, layerY_hidden2, centerY) {
  // Input → Hidden1
  for (let i = 0; i < 3; i++) {
    drawArrow(
      layerX[0] + 30, centerY,
      layerX[1] - 30, layerY_hidden1[i],
      false, "#ccc",
      map(Math.abs(w1[i]), 0, 0.5, 1, 4)
    );
  }
  // Hidden1 → Hidden2
  for (let j = 0; j < 2; j++) {
    for (let i = 0; i < 3; i++) {
      drawArrow(
        layerX[1] + 30, layerY_hidden1[i],
        layerX[2] - 30, layerY_hidden2[j],
        false, "#ccc",
        map(Math.abs(w2[j][i]), 0, 0.5, 1, 4)
      );
    }
  }
  // Hidden2 → Output
  for (let j = 0; j < 2; j++) {
    drawArrow(
      layerX[2] + 30, layerY_hidden2[j],
      layerX[3] - 30, centerY,
      false, "#ccc",
      map(Math.abs(w3[j]), 0, 0.5, 1, 4)
    );
  }
}

function highlightConnection(conn, layerX, layerY_hidden1, layerY_hidden2, centerY) {
  let color = conn.color || "#0077cc"; // blue default
  if (currentPhase === "backward") color = "red";

  if (conn.from === "x") {
    const i = parseInt(conn.to.replace("h1", "")) - 1;
    drawArrow(
      layerX[0] + 30, centerY,
      layerX[1] - 30, layerY_hidden1[i],
      true, color,
      map(Math.abs(w1[i]), 0, 0.5, 1, 4)
    );
  }
  else if (conn.from.startsWith("h1")) {
    const i = parseInt(conn.from.replace("h1", "")) - 1;
    const j = parseInt(conn.to.replace("h2", "")) - 1;
    drawArrow(
      layerX[1] + 30, layerY_hidden1[i],
      layerX[2] - 30, layerY_hidden2[j],
      true, color,
      map(Math.abs(w2[j][i]), 0, 0.5, 1, 4)
    );
  }
  else if (conn.from.startsWith("h2") && conn.to === "yhat") {
    const j = parseInt(conn.from.replace("h2", "")) - 1;
    drawArrow(
      layerX[2] + 30, layerY_hidden2[j],
      layerX[3] - 30, centerY,
      true, color,
      map(Math.abs(w3[j]), 0, 0.5, 1, 4)
    );
  }
  else if (conn.from === "yhat") {
    // backward: output → hidden2
    const j = parseInt(conn.to.replace("h2", "")) - 1;
    drawArrow(
      layerX[3] - 30, centerY,
      layerX[2] + 30, layerY_hidden2[j],
      true, color,
      map(Math.abs(w3[j]), 0, 0.5, 1, 4)
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
  push();
  const arrowColor = colorOverride || (active ? "#0077cc" : "#aaa");
  stroke(arrowColor);
  strokeWeight(thickness);
  noFill();
  line(x1, y1, x2, y2);

  const angle = atan2(y2 - y1, x2 - x1);
  translate(x2, y2);
  rotate(angle);

  fill(arrowColor);
  noStroke();
  triangle(0, 0, -10, 5, -10, -5);
  pop();
}

function drawLossGraph() {
  const graphX = 50, graphY = 500, graphW = 700, graphH = 70;
  const maxLoss = Math.max(...lossHistory.map(p => p.loss), 1);

  stroke(0);
  noFill();
  rect(graphX, graphY, graphW, graphH);

  noStroke();
  fill(0);
  textSize(12);
  text("Loss", graphX - 20, graphY + graphH / 2);
  text("Epoch", graphX + graphW / 2, graphY + graphH + 15);

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
