import { useState, useEffect, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgpu";
import Loader from "./components/loader";
import ButtonHandler from "./components/btn-handler";
import { detectPose, detectVideo } from "./utils/detect"; // Import detectPose
import "./style/App.css";

tf.setBackend("webgpu"); // set backend to webgpu

/**
 * This component initializes and loads a pose detection model using TensorFlow.js,
 * sets up references for image, camera, video, and canvas elements, and
 * handles the loading state and model configuration.
 */

const App = () => {
  const [loading, setLoading] = useState({ loading: true, progress: 0 }); // loading state
  const [model, setModel] = useState({
    net: null,
    inputShape: [1, 0, 0, 3],
  }); // init model & input shape

  // references
  const imageRef = useRef(null);
  const cameraRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  // model configs
  const modelName = "yolo11n";

  useEffect(() => {
    tf.ready().then(async () => {
      const yolo11 = await tf.loadGraphModel(
        `${window.location.href}/${modelName}_web_model/model.json`,
        {
          onProgress: (fractions) => {
            setLoading({ loading: true, progress: fractions }); // set loading fractions
          },
        }
      ); // load model

      // warming up model
      const dummyInput = tf.ones(yolo11.inputs[0].shape);
      const warmupResults = yolo11.execute(dummyInput);

      setLoading({ loading: false, progress: 1 });
      setModel({
        net: yolo11,
        inputShape: yolo11.inputs[0].shape,
      }); // set model & input shape

      tf.dispose([warmupResults, dummyInput]); // cleanup memory
    });
  }, []);

  const handleVideoLoadedData = () => {
    if (cameraRef.current && cameraRef.current.readyState >= 2) {
      detectPose(cameraRef.current, model, canvasRef.current).catch((error) => {
        console.error("Error during pose detection:", error);
      });
    }
  };

  return (
    <div className="App">
      {loading.loading && (
        <Loader>Loading model... {(loading.progress * 100).toFixed(2)}%</Loader>
      )}
      <div className="header">
        <h1>📷 Pose Detection App</h1>
        <p>
          Pose detection application on browser powered by{" "}
          <code>tensorflow.js</code>
        </p>
        <p>
          Serving : <code className="code">{modelName}</code>
        </p>
      </div>

      <div className="content">
        <img
          src="#"
          ref={imageRef}
          onLoad={() => detectPose(imageRef.current, model, canvasRef.current).catch((error) => {
            console.error("Error during pose detection:", error);
          })}
        />
        <video
          autoPlay
          muted
          ref={cameraRef}
          onLoadedData={handleVideoLoadedData}
        />
        <ButtonHandler
          imageRef={imageRef}
          cameraRef={cameraRef}
          videoRef={videoRef}
        />
      </div>
    </div>
  );
};

export default App;