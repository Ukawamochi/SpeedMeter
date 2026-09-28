// onnxruntime-web の読み込みとモデルのセッション（index.html・check.html で共通）。
// CDN（cdn.jsdelivr.net）から版を固定して読む。wasm（CPU）、1スレッド（cross-origin isolation 不要）。

import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.mjs';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
ort.env.wasm.numThreads = 1;

export { ort };
// scripts/setup_web_model.py が置く。一覧は [{ id, file, sha256 }]（--source の順）
export const MODELS_DIR = 'models/';
export const MANIFEST_URL = `${MODELS_DIR}models.json`;

export async function loadManifest() {
  const response = await fetch(MANIFEST_URL);
  if (!response.ok) throw new Error(`${MANIFEST_URL} が読めない（scripts/setup_web_model.py を実行する）`);
  return response.json();
}

export function createSession(file) {
  return ort.InferenceSession.create(`${MODELS_DIR}${file}`, { executionProviders: ['wasm'] });
}
