let x_vals = [];
let y_vals = [];

let coefficients = [];

let degree = 3;
let learningRate = 0.5

let tensorCounter;

const optimizer = tf.train.sgd(learningRate)

function setup() {
  initCoeficients()

  // Render canvas and html stuff
  canvas = createCanvas(800, 600);
  canvas.parent('sketch-holder');
  canvas.mousePressed(addPoints);
  bindHTML()
}

function bindHTML() {
  tensorCounter = document.getElementById('tensorsCounter');
  tensorContainers = document.getElementById('tensorContainers');
  learningRateInput = document.getElementById('learningRateInput');
  degreeInput = document.getElementById('degreeInput')  
}

function initCoeficients() {
  for(let i = 0; i<degree; i++) {
    coefficients.push(tf.variable(tf.scalar(random(1)), true))
  }
}

// Train the model.
function train() {
  tf.tidy(() => {
    optimizer.minimize(() =>  {
      const labels = tf.tensor1d(y_vals);
      return loss(labels, predict(x_vals))
    }, true, coefficients);
  })
}

function predict(x) {
  const xs = tf.tensor1d(x);
  let ys = tf.variable(tf.zerosLike(xs));
  for (let i = 0; i < degree; i++) {
    const coef = coefficients[i];
    const pow_ts = tf.fill(xs.shape, degree - i);
    const sum = tf.add(ys, coefficients[i].mul(xs.pow(pow_ts)));
    ys.dispose();
    ys = sum.clone();
  }
  return ys;
}

function loss(labels, prediction) {
  return labels.sub(prediction).square().mean();
}

function addPoints() {
  let x = map(mouseX, 0, width, -1, 1);
  let y = map(mouseY, 0, height, -1, 1);
  x_vals.push(x);
  y_vals.push(y);
}

function draw() {

  background(0);

  stroke(255);
  strokeWeight(8);

  for (let i = 0; i < x_vals.length; i++) {
    let px = map(x_vals[i], -1, 1, 0, width);
    let py = map(y_vals[i], -1, 1, 0, height);
    point(px, py);
  }

  if(x_vals.length) {
    train();
  }
  drawFunctionShape();
  displayInformation();
}

function drawFunctionShape() {
  const curveX = [];
  for (let x = -1; x <= 1; x += 0.05) {
    curveX.push(x);
  }

  const ys = tf.tidy(() => predict(curveX))
  let curveY = ys.dataSync();
  ys.dispose();

  beginShape();
  noFill();
  stroke(255);
  strokeWeight(2);
  for (let i = 0; i < curveX.length; i++) {
    let x = map(curveX[i], -1, 1, 0, width);
    let y = map(curveY[i], -1, 1, 0, height);
    vertex(x, y);
  }
  endShape();
}

function displayInformation(){
  tensorCounter.innerHTML = tf.memory().numTensors;
  tensorContainers.innerHTML = '';
  for(let i = 0; i<degree; i++) {
    tensorContainers.append(` 

      Tensor: ${i} value:  ${coefficients[i].dataSync()[0]}

  `)
  }
}

function resetModel() {
  initCoeficients();
  degree = parseInt(degreeInput.value);
  learningRate = parseFloat(learningRateInput.value);
}

function clearData(){
  x_vals = [];
  y_vals = [];
}