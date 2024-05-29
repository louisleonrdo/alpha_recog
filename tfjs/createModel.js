import * as tf from '@tensorflow/tfjs';
// import * as tf from '@tensorflow/tfjs';

// Load the MNIST dataset

// Convert the dataset into tensors
// const { xs, ys } = await mnist;
const numClasses = 10; // Number of classes (digits 0-9)

// Convert the labels to one-hot encoding
const ysOneHot = tf.oneHot(ys, numClasses);

// Normalize the pixel values (between 0 and 1)
const xsNormalized = xs.div(255);

// Split the dataset into training and testing sets
const split = 0.8; // Percentage of data for training
const numExamples = xs.shape[0];
const numTrainExamples = Math.floor(numExamples * split);

const xTrain = xsNormalized.slice([0], [numTrainExamples]);
const yTrain = ysOneHot.slice([0], [numTrainExamples]);
const xTest = xsNormalized.slice([numTrainExamples], [numExamples]);
const yTest = ysOneHot.slice([numTrainExamples], [numExamples]);

// Now you can use xTrain, yTrain, xTest, yTest for training and testing your model

// Define your model
const model = tf.sequential();
model.add(tf.layers.conv2d({
  inputShape: [28, 28, 1],
  kernelSize: 3,
  filters: 16,
  activation: 'relu',
}));
model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
model.add(tf.layers.conv2d({ kernelSize: 3, filters: 32, activation: 'relu' }));
model.add(tf.layers.flatten());
model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

// Compile your model
model.compile({
  optimizer: 'adam',
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy'],
});

// Train your model
const epochs = 10;
await model.fit(xTrain, yTrain, { epochs, validationSplit: 0.1 });

// Evaluate your model
const evalResult = model.evaluate(xTest, yTest);
console.log('Test accuracy:', evalResult[1].dataSync());

// Save your model
await model.save('model');
