const x_vals = [];
const y_vals = [];

let bias;
let weight;
let bias_data;
let weight_data;
const learningRate = 0.5
const optimizer = tf.train.sgd(learningRate)


function setup() {
  createCanvas(800, 600);

  bias = tf.scalar(random(1)).variable();
  weight = tf.scalar(random(1)).variable();

  bias_data = bias.dataSync()[0]
  weight_data = weight.dataSync()[0]

}

// Train the model.
function train() {
  tf.tidy(() => {
    optimizer.minimize(() =>  {
      const labels = tf.tensor1d(y_vals);
      const xs = tf.tensor1d(x_vals);
      return loss(labels, predict(xs))
    });
  })
}

// Linear Regression
function predict(x) {
  // y = m * x + b
  return x.mul(weight).add(bias);
}

function loss(labels, prediction) {
  return labels.sub(prediction).square().mean();
}


function drawGradient() {
  line(0, 0, height, weight);
};

function mousePressed() {
  let x = map(mouseX, 0, width, 0, 1);
  let y = map(mouseY, 0, height, 1, 0);
  x_vals.push(x);
  y_vals.push(y);
}

function draw() {

  background(0);

  stroke(255);
  strokeWeight(8);

  for (let i = 0; i < x_vals.length; i++) {
    let px = map(x_vals[i], 0, 1, 0, width);
    let py = map(y_vals[i], 0, 1, height, 0);
    point(px, py);
  }

  if(x_vals.length) {
    train();
    weight_data = weight.dataSync()[0];
    bias_data = bias.dataSync()[0];
  }

  console.log(tf.memory().numTensors);
  console.log(weight.dataSync()[0]);
  console.log(bias.dataSync()[0])
  drawGradient();

}

function slope(x) {
  return  x * weight_data + bias_data;
}

function drawGradient() {
  let x1 = 0;
  let y1 = slope(x1);
  let x2 = 1;
  let y2 = slope(x2);

  let denormX1 = Math.floor(map(0, 0, 1, 0, width))
  let denormY1 = Math.floor(map(y1, 0, 1, 0, height))
  let denormX2 = Math.floor(map(1, 0, 1, 0, width))
  let denormY2 = Math.floor(map(y2, 0, 1, 0, height))

  stroke(255);
  line(denormX1, denormY1, denormX2, denormY2);
}
