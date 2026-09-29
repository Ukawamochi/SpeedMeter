// 参照入力（web/reference/*.json の waveform_int16）から、ページと同じ関数（web/dsp.js）で
// ONNX の入力を作り、float32 のバイナリで書き出す。tests/test_web_demo.py が ONNX Runtime
// （Python）に通して参照値の mora と比べる（onnxruntime-node を入れずに JS の経路を確かめるため）。
//
// 使い方: node web/tests/dump_reference_logmel.mjs <reference.json> <out_prefix>
//   <out_prefix>.window.f32 / <out_prefix>.whole.f32（リトルエンディアンの float32 の並び）と
//   <out_prefix>.json（形と窓の開始位置）を書く。

import { readFileSync, writeFileSync } from 'node:fs';
import { buildWholeInput, buildWindowInput } from '../dsp.js';

const [referencePath, outPrefix] = process.argv.slice(2);
if (!referencePath || !outPrefix) {
  console.error('usage: node dump_reference_logmel.mjs <reference.json> <out_prefix>');
  process.exit(2);
}
const reference = JSON.parse(readFileSync(referencePath, 'utf-8'));
const samples = Float32Array.from(reference.waveform_int16, (v) => Math.fround(v / 32768));
const windows = buildWindowInput(samples);
const whole = buildWholeInput(samples);
const bytes = (arr) => Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength);
writeFileSync(`${outPrefix}.window.f32`, bytes(windows.data));
writeFileSync(`${outPrefix}.whole.f32`, bytes(whole.data));
writeFileSync(`${outPrefix}.json`, JSON.stringify({ window_dims: windows.dims, window_starts: windows.starts, whole_dims: whole.dims }));
