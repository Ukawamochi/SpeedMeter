// 計算の照合ページ（確認用）。参照入力をブラウザの経路（dsp.js → onnxruntime-web）で処理し、
// Python の経路の出力（scripts/make_web_reference.py が書いた参照値）との差を表示する。
// 入力は2通り: docs/spec.md の窓（2.0 秒、0.25 秒ずらし）と、確認用のクリップ全体1回（仕様外）。

import { buildWholeInput, buildWindowInput } from './dsp.js';
import { createSession, ort } from './model.js';

const REFERENCE_URL = 'reference/onnx_exp005_fp32.json';
const MODEL_FILE = 'exp005.onnx'; // 参照値を作ったモデル（scripts/setup_web_model.py が置く）
const REFERENCE_ATOL = 1e-3; // モーラ。tests/test_web_demo.py の MORA_ATOL と同じ

const statusEl = document.getElementById('status');
const resultEl = document.getElementById('result');
const button = document.getElementById('run');

function block(title, obj) {
  const h = document.createElement('h2');
  h.textContent = title;
  const pre = document.createElement('pre');
  pre.textContent = typeof obj === 'string' ? obj : JSON.stringify(obj, null, 1);
  resultEl.append(h, pre);
}

async function runAndCompare(session, name, input, expected) {
  const outputs = await session.run({ log_mel: new ort.Tensor('float32', input.data, input.dims) });
  const got = Array.from(outputs.mora.data);
  const maxDiff = Math.max(...got.map((v, i) => Math.abs(v - expected[i])));
  block(name, {
    input_dims: input.dims,
    output_dims: outputs.mora.dims,
    browser: got,
    python: expected,
    max_abs_diff: maxDiff,
    judge: maxDiff <= REFERENCE_ATOL ? `OK（${REFERENCE_ATOL} モーラ以下）` : `超過（目安 ${REFERENCE_ATOL} モーラ）`,
  });
}

button.addEventListener('click', async () => {
  button.disabled = true;
  resultEl.replaceChildren();
  try {
    statusEl.textContent = '読み込み中…';
    const [reference, session] = await Promise.all([
      fetch(REFERENCE_URL).then((r) => r.json()),
      createSession(MODEL_FILE),
    ]);
    const samples = Float32Array.from(reference.waveform_int16, (v) => Math.fround(v / 32768));
    statusEl.textContent = '計算中…';
    await runAndCompare(session, '窓ごと（2.0 秒窓・0.25 秒ずらし）', buildWindowInput(samples), reference.window_mora);
    await runAndCompare(session, '確認用: クリップ全体を1回入力（仕様外）', buildWholeInput(samples), reference.whole_mora);
    statusEl.textContent = `完了（onnxruntime-web ${ort.env.versions.web}、参照値は onnxruntime ${reference.onnxruntime}）`;
  } catch (error) {
    statusEl.textContent = `エラー: ${error}`;
    console.error(error);
  } finally {
    button.disabled = false;
  }
});
