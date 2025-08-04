let step = 0;
let timer = 0;
let stepText = "";

function setup() {
  const canvas = createCanvas(400, 300);
  canvas.parent('sketch-holder');
  textAlign(CENTER, CENTER);
  textSize(14);
}

function draw() {
  background(255);
  drawNetwork();
  animateFlow();
  document.getElementById("ann-step-desc").innerText = stepText;
}

function drawNeuron(x, y, label) {
  fill(255);
  stroke(0);
  ellipse(x, y, 40);
  fill(0);
  noStroke();
  text(label, x, y);
}

function drawArrow(x1, y1, x2, y2, active) {
  stroke(active ? '#0077cc' : '#aaa');
  strokeWeight(active ? 3 : 1);
  line(x1, y1, x2, y2);
  push();
  const angle = atan2(y2 - y1, x2 - x1);
  translate(x2, y2);
  rotate(angle);
  fill(active ? '#0077cc' : '#aaa');
  triangle(0, 0, -10, 5, -10, -5);
  pop();
}

function drawNetwork() {
  // Input
  drawNeuron(50, 100, "x₁");
  drawNeuron(50, 200, "x₂");

  // Hidden
  drawNeuron(200, 100, "h₁");
  drawNeuron(200, 200, "h₂");

  // Output
  drawNeuron(350, 150, "ŷ");

  // Arrows
  drawArrow(70, 100, 180, 100, step === 1);
  drawArrow(70, 200, 180, 100, step === 2);
  drawArrow(70, 100, 180, 200, step === 3);
  drawArrow(70, 200, 180, 200, step === 4);
  drawArrow(220, 100, 330, 150, step === 5);
  drawArrow(220, 200, 330, 150, step === 6);
}

function animateFlow() {
  timer++;
  if (timer % 60 === 0) {
    step = (step + 1) % 7;
  }

  const messages = [
    "Start of forward pass",
    "x₁ → h₁",
    "x₂ → h₁",
    "x₁ → h₂",
    "x₂ → h₂",
    "h₁ → ŷ",
    "h₂ → ŷ"
  ];
  stepText = messages[step];
}
