// onnxruntime-web（同梱。vendor/ は scripts/build_extension.py が置く）。拡張機能は外部の CDN を読めないので、
// wasm もこの拡張機能のファイルから読む。1スレッド（cross-origin isolation 不要）。
import * as ort from '../vendor/ort.wasm.min.mjs';

ort.env.wasm.wasmPaths = new URL('../vendor/', import.meta.url).href;
ort.env.wasm.numThreads = 1;

export { ort };
