import * as tf from '@tensorflow/tfjs';

/**
 * Function to preprocess the image for pose detection.
 * @param {HTMLImageElement|HTMLVideoElement} source
 * @param {number} modelWidth
 * @param {number} modelHeight
 * @returns {tf.Tensor} preprocessed input tensor
 */
const preprocess = (source, modelWidth, modelHeight) => {
  const input = tf.browser.fromPixels(source)
    .resizeBilinear([modelWidth, modelHeight])
    .expandDims(0)
    .toFloat()
    .div(255.0); // normalize

  return input;
};

/**
 * Function to run inference and do pose detection from source.
 * @param {HTMLImageElement|HTMLVideoElement} source
 * @param {tf.GraphModel} model loaded pose detection tensorflow.js model
 * @param {HTMLCanvasElement} canvasRef canvas reference
 * @param {VoidFunction} callback function to run after detection process
 */
export const detectPose = async (source, model, canvasRef, callback = () => {}) => {
  const [modelWidth, modelHeight] = model.inputShape.slice(1, 3); // get model width and height

  tf.engine().startScope(); // start scoping tf engine

  const input = preprocess(source, modelWidth, modelHeight); // preprocess image

  const pose = await model.net.estimateSinglePose(input, {
    flipHorizontal: false,
  });

  // Draw pose on canvas
  const ctx = canvasRef.getContext('2d');
  ctx.clearRect(0, 0, canvasRef.width, canvasRef.height);
  ctx.drawImage(source, 0, 0, canvasRef.width, canvasRef.height);

  pose.keypoints.forEach((keypoint) => {
    if (keypoint.score > 0.5) {
      const { y, x } = keypoint.position;
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, 2 * Math.PI);
      ctx.fillStyle = 'red';
      ctx.fill();
    }
  });

  tf.engine().endScope(); // end scoping tf engine

  callback(); // run callback function
};

/**
 * Function to detect pose in each frame of the video.
 * @param {HTMLVideoElement} video
 * @param {tf.GraphModel} model
 * @param {HTMLCanvasElement} canvasRef
 */
export const detectVideo = (video, model, canvasRef) => {
  const detectFrame = async () => {
    await detectPose(video, model, canvasRef);
    requestAnimationFrame(detectFrame);
  };

  detectFrame(); // initialize to detect every frame
};

