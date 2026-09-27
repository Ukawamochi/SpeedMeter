// 確認用ページ（ユーザーの指示による、ブラウザで動作を確かめるための最小限のページ）。
// 仕様（docs/spec.md）の実装ではない。表示の仕方・入力の組み方のうち仕様に無いもの
// （クリップ全体を1回入力する表示）は「確認用」であり、仕様を決めたものではない。
//
// 流れ: マイク（getUserMedia）→ AudioWorklet でブラウザの標本化周波数のまま録音（チャンネル平均で
// モノラル）→ 停止後に 16kHz へ再標本化（web/dsp.js resampleSinc: hann 窓付き sinc 補間。
// torchaudio の既定 sinc_interp_hann と同じ核。Python 側の librosa/soxr とは別の方法）
// → 対数メル（web/dsp.js）→ onnxruntime-web（wasm、CPU）で model_fp32.onnx を実行。

import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.mjs';
import {
  MEL_CONFIG,
  WINDOW_SEC,
  buildWholeInput,
  buildWindowInput,
  parseWholeOutput,
  parseWindowOutputs,
  resampleSinc,
} from './dsp.js';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
ort.env.wasm.numThreads = 1; // SharedArrayBuffer（cross-origin isolation）を要しない設定

const MODEL_URL = 'models/model_fp32.onnx';
const REFERENCE_URL = 'reference/onnx_exp005_fp32.json';
const TARGET_RATE = MEL_CONFIG.sampleRate; // 16000
const REFERENCE_ATOL = 1e-3; // 参照値との差の目安（モーラ）。tests/test_web_demo.py の MORA_ATOL と同じ

const recordButton = document.getElementById('record');
const referenceButton = document.getElementById('reference');
const statusEl = document.getElementById('status');
const resultEl = document.getElementById('result');

let sessionPromise = null;
function getSession() {
  if (sessionPromise === null) {
    sessionPromise = ort.InferenceSession.create(MODEL_URL, { executionProviders: ['wasm'] });
  }
  return sessionPromise;
}

function setStatus(text) {
  statusEl.textContent = text;
}

function el(tag, text) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  return node;
}

function pre(obj) {
  return el('pre', typeof obj === 'string' ? obj : JSON.stringify(obj, null, 1));
}

function fmt(x, digits = 4) {
  return Number(x).toFixed(digits);
}

// ---- 録音 ------------------------------------------------------------------------------

let recording = null;

async function startRecording() {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
  });
  const context = new AudioContext(); // 既定（ブラウザ・機器の標本化周波数）のまま録る
  await context.audioWorklet.addModule('recorder-worklet.js');
  const source = context.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(context, 'recorder-processor');
  const mute = context.createGain();
  mute.gain.value = 0; // 処理を駆動するため出力へつなぐが、音は出さない
  const chunks = [];
  node.port.onmessage = (event) => chunks.push(event.data);
  source.connect(node).connect(mute).connect(context.destination);
  const settings = stream.getAudioTracks()[0].getSettings();
  recording = { stream, context, source, node, chunks, settings };
  recordButton.textContent = '録音停止';
  setStatus(`録音中（${context.sampleRate} Hz）…`);
}

async function stopRecording() {
  const { stream, context, source, node, chunks, settings } = recording;
  recording = null;
  source.disconnect();
  node.disconnect();
  stream.getTracks().forEach((t) => t.stop());
  const inputRate = context.sampleRate;
  await context.close();
  recordButton.textContent = '録音開始';

  const total = chunks.reduce((n, c) => n + c.length, 0);
  const raw = new Float32Array(total);
  let offset = 0;
  for (const c of chunks) {
    raw.set(c, offset);
    offset += c.length;
  }
  setStatus('16kHz へ再標本化中…');
  const samples = resampleSinc(raw, inputRate, TARGET_RATE);
  await analyze(samples, {
    source: 'マイク',
    inputRate,
    inputSamples: raw.length,
    trackSettings: settings,
    resample: inputRate === TARGET_RATE
      ? 'なし（入力が 16kHz）'
      : `${inputRate} Hz → ${TARGET_RATE} Hz、hann 窓付き sinc 補間（lowpass_filter_width=6、rolloff=0.99）`,
  });
}

recordButton.addEventListener('click', async () => {
  recordButton.disabled = true;
  try {
    if (recording === null) await startRecording();
    else await stopRecording();
  } catch (error) {
    setStatus(`エラー: ${error}`);
    console.error(error);
    recording = null;
    recordButton.textContent = '録音開始';
  } finally {
    recordButton.disabled = false;
  }
});

// ---- 参照入力（Python の出力との照合） ---------------------------------------------------

referenceButton.addEventListener('click', async () => {
  referenceButton.disabled = true;
  try {
    setStatus('参照入力を読み込み中…');
    const reference = await (await fetch(REFERENCE_URL)).json();
    const samples = Float32Array.from(reference.waveform_int16, (v) => Math.fround(v / 32768));
    await analyze(samples, {
      source: `参照入力（${REFERENCE_URL}。合成波形 3.0 秒、16kHz）`,
      inputRate: reference.sample_rate,
      inputSamples: samples.length,
      resample: 'なし',
      reference,
    });
  } catch (error) {
    setStatus(`エラー: ${error}`);
    console.error(error);
  } finally {
    referenceButton.disabled = false;
  }
});

// ---- 推論と表示 --------------------------------------------------------------------------

async function run(session, input) {
  const tensor = new ort.Tensor('float32', input.data, input.dims);
  const started = performance.now();
  const outputs = await session.run({ log_mel: tensor });
  const elapsedMs = performance.now() - started;
  const out = outputs.mora;
  return { out, elapsedMs };
}

function rawOutput(inputDims, out, elapsedMs) {
  return pre({
    input: { name: 'log_mel', dims: inputDims },
    output: { name: 'mora', type: out.type, dims: out.dims, data: Array.from(out.data) },
    session_run_ms: Number(elapsedMs.toFixed(1)),
  });
}

function referenceDiff(got, expected) {
  const diffs = Array.from(got, (v, i) => Math.abs(v - expected[i]));
  const max = Math.max(...diffs);
  return `参照値（Python の対数メル → ONNX Runtime CPU）: ${JSON.stringify(expected)}\n`
    + `max|差| = ${max.toExponential(3)} モーラ（目安 ${REFERENCE_ATOL} 以下: ${max <= REFERENCE_ATOL ? 'OK' : '超過'}）`;
}

function wavBlob(samples, rate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeStr = (o, s) => [...s].forEach((c, i) => view.setUint8(o + i, c.charCodeAt(0)));
  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeStr(8, 'WAVEfmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  samples.forEach((v, i) => view.setInt16(44 + i * 2, Math.max(-32768, Math.min(32767, Math.round(v * 32767))), true));
  return new Blob([buffer], { type: 'audio/wav' });
}

async function analyze(samples, info) {
  resultEl.replaceChildren();
  const durationSec = samples.length / TARGET_RATE;

  resultEl.append(el('h2', '入力'));
  resultEl.append(pre({
    source: info.source,
    input_sample_rate_hz: info.inputRate,
    input_samples: info.inputSamples,
    resample: info.resample,
    model_sample_rate_hz: TARGET_RATE,
    samples_16k: samples.length,
    duration_sec: Number(durationSec.toFixed(4)),
    ...(info.trackSettings ? { track_settings: info.trackSettings } : {}),
  }));
  const audio = el('audio');
  audio.controls = true;
  audio.src = URL.createObjectURL(wavBlob(samples, TARGET_RATE));
  resultEl.append(el('p', '16kHz に変換後の音声:'), audio);

  setStatus('モデルを読み込み中…');
  const session = await getSession();
  setStatus('対数メルを計算・推論中…');

  // 1) 窓ごと（docs/spec.md の窓: 2.0 秒、0.25 秒ずらし、窓全体が収まるものだけ）
  resultEl.append(el('h2', `窓ごと（${WINDOW_SEC} 秒窓・0.25 秒ずらし。docs/spec.md の窓）`));
  const windowInput = buildWindowInput(samples);
  if (windowInput.starts.length === 0) {
    resultEl.append(el('p', `録音が ${WINDOW_SEC} 秒未満のため窓が無い（窓全体が収まるものだけを取る）。`));
  } else {
    const { out, elapsedMs } = await run(session, windowInput);
    resultEl.append(el('h3', 'モデルの返すナマの値'), rawOutput(windowInput.dims, out, elapsedMs));
    if (info.reference) resultEl.append(pre(referenceDiff(out.data, info.reference.window_mora)));
    const parsed = parseWindowOutputs(out.data, windowInput.starts);
    const table = el('table');
    const head = el('tr');
    ['窓', '区間 [秒]', 'モーラ数', '毎秒モーラ数（÷2.0秒）'].forEach((h) => head.append(el('th', h)));
    table.append(head);
    parsed.forEach((row, i) => {
      const tr = el('tr');
      [String(i), `${fmt(row.startSec, 2)}–${fmt(row.endSec, 2)}`, fmt(row.mora), fmt(row.moraPerSec)]
        .forEach((v) => tr.append(el('td', v)));
      table.append(tr);
    });
    resultEl.append(el('h3', 'パースした値'), table);
  }

  // 2) 確認用: クリップ全体を1回入力（仕様に無い入力。比較のための表示）
  resultEl.append(el('h2', '確認用: クリップ全体を1回入力（仕様外）'));
  if (samples.length <= MEL_CONFIG.nFft / 2) {
    resultEl.append(el('p', '録音が短すぎる（reflect の詰め物に 201 標本以上が必要）。'));
  } else {
    const wholeInput = buildWholeInput(samples);
    const { out, elapsedMs } = await run(session, wholeInput);
    resultEl.append(el('h3', 'モデルの返すナマの値'), rawOutput(wholeInput.dims, out, elapsedMs));
    if (info.reference) resultEl.append(pre(referenceDiff(out.data, info.reference.whole_mora)));
    const parsed = parseWholeOutput(out.data[0], samples.length);
    resultEl.append(el('h3', 'パースした値'), pre(
      `総モーラ数: ${fmt(parsed.mora)}\n`
      + `録音の長さ: ${fmt(parsed.durationSec)} 秒\n`
      + `毎秒モーラ数（モーラ数 ÷ 録音の秒数）: ${fmt(parsed.moraPerSec)}`,
    ));
  }
  setStatus('完了');
}
