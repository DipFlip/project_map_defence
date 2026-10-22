// Hand tracking off the main thread: game.js posts camera frames (ImageBitmaps), this
// worker runs MediaPipe on them and posts the landmarks back, so a slow detection never
// holds up drawing the ball.
import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.1/vision_bundle.mjs";

let landmarker = null;

// MediaPipe loads its WASM glue with importScripts, which module workers don't have; it then
// falls back to self.import(url). The glue is a classic script that defines a global
// ModuleFactory, so run it in the worker's global scope.
self.import = async (url) => {
  const src = await (await fetch(url)).text();
  (0, eval)(`${src}\n//# sourceURL=${url}`);
};

onmessage = async (e) => {
  const m = e.data;
  if (m.type === "init") {
    try {
      const vision = await FilesetResolver.forVisionTasks(m.wasm);
      let used;
      for (const delegate of ["GPU", "CPU"]) {
        try {
          landmarker = await HandLandmarker.createFromOptions(vision, { ...m.options, baseOptions: { modelAssetPath: m.model, delegate } });
          used = delegate;
          break;
        } catch (err) {
          if (delegate === "CPU") throw err;
        }
      }
      postMessage({ type: "ready", delegate: used });
    } catch (err) {
      postMessage({ type: "error", message: String(err && err.message || err) });
    }
    return;
  }
  // a frame
  try {
    const t0 = performance.now();
    const res = landmarker.detectForVideo(m.bitmap, m.t);
    postMessage({ type: "hands", t: m.t, ms: performance.now() - t0, hands: res.landmarks.map((lm) => lm.map((p) => [p.x, p.y])) });
  } catch (err) {
    postMessage({ type: "error", message: String(err && err.message || err) });
  } finally {
    m.bitmap.close();
  }
};
