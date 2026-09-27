// onnxruntime-web の読み込みとモデルのセッション（index.html・check.html で共通）。
// CDN（cdn.jsdelivr.net）から版を固定して読む。wasm（CPU）、1スレッド（cross-origin isolation 不要）。

import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.mjs';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
ort.env.wasm.numThreads = 1;

export { ort };
export const MODEL_URL = 'models/model_fp32.onnx'; // scripts/setup_web_model.py が置く

export function createSession() {
  return ort.InferenceSession.create(MODEL_URL, { executionProviders: ['wasm'] });
}
