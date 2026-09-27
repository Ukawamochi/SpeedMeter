// 確認用ページ（ユーザーの指示による、ブラウザで動作を確かめるための最小限のページ）。
// 仕様（docs/spec.md）の実装ではない。
//
// 録音中は推論を動かし続ける（docs/spec.md: 2.0 秒窓、推論間隔 0.25 秒、表示は出力 ÷ 窓長）。
//   マイク（getUserMedia）→ AudioWorklet でブラウザの標本化周波数のまま取り（チャンネル平均でモノラル）
//   → 環形バッファ（直近 4 秒）→ 0.25 秒ごとに直近 2.0 秒を切り出して 16kHz へ変換（web/dsp.js resampleSinc:
//   hann 窓付き sinc 補間。Python 側の librosa/soxr とは別の方法）→ 対数メル → onnxruntime-web
//   → 毎秒モーラ数 = モーラ数 ÷ 2.0 秒。推論が次の時点に間に合わなければその時点は間引く（web/realtime.js）。
// 早口の閾値は仕様に無いので決めていない（docs/questions.md）。グラフには評価の話速帯の境界（4・6・8）を
// 薄く引くだけにする。値は平滑化せずそのまま描く。

import { WINDOW_SEC, logMelSpectrogram } from './dsp.js';
import { createSession, ort } from './model.js';
import {
  BAND_BOUNDARIES,
  InferenceScheduler,
  RingBuffer,
  extractWindow16k,
  makeScale,
  pruneHistory,
  yAxisMax,
} from './realtime.js';

const SPAN_SEC = 30; // グラフの横軸（直近の秒数）
const RING_SEC = 4; // 環形バッファの長さ（2.0 秒窓 + 余白より十分長く）

const recordButton = document.getElementById('record');
const statusEl = document.getElementById('status');
const currentEl = document.getElementById('current');
const statsEl = document.getElementById('stats');
const canvas = document.getElementById('chart');

let sessionPromise = null;
let rec = null; // 録音中の状態
let history = []; // { t: 窓の終わりの時刻（録音開始からの秒）, v: 毎秒モーラ数 }
let nowSec = 0;

function getSession() {
  if (sessionPromise === null) sessionPromise = createSession();
  return sessionPromise;
}

// ---- 推論 ------------------------------------------------------------------------------

async function inferAt(state, end) {
  const started = performance.now();
  const window16k = extractWindow16k(state.ring, end, state.rate);
  if (window16k === null) return;
  const mel = logMelSpectrogram(window16k); // (201, 80)
  const input = new ort.Tensor('float32', mel.data, [1, mel.numFrames, mel.nMels]);
  const outputs = await state.session.run({ log_mel: input });
  const mora = outputs.mora.data[0];
  const moraPerSec = mora / WINDOW_SEC;
  state.lastMs = performance.now() - started;
  if (rec !== state) return; // 停止後に終わった推論は表示しない
  const t = end / state.rate;
  history.push({ t, v: moraPerSec });
  nowSec = t;
  history = pruneHistory(history, nowSec, SPAN_SEC);
  currentEl.firstChild.textContent = `${moraPerSec.toFixed(2)} `;
  draw();
  showStats(state);
}

function onChunk(state, chunk) {
  state.ring.push(chunk);
  const end = state.scheduler.update(state.ring.totalWritten, state.busy);
  if (end === null) {
    showStats(state);
    return;
  }
  state.busy = true;
  inferAt(state, end)
    .catch((error) => {
      statusEl.textContent = `推論のエラー: ${error}`;
      console.error(error);
    })
    .finally(() => {
      state.busy = false;
    });
}

function showStats(state) {
  const s = state.scheduler;
  statsEl.textContent = `入力 ${state.rate} Hz → 16000 Hz ／ 推論 ${s.runs} 回 ／ 間引き ${s.skipped} 回`
    + (state.lastMs !== undefined ? ` ／ 直近の処理 ${state.lastMs.toFixed(0)} ms` : '');
}

// ---- 録音 ------------------------------------------------------------------------------

async function start() {
  statusEl.textContent = 'モデルを読み込み中…';
  const session = await getSession();
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
  });
  const context = new AudioContext(); // ブラウザ・機器の標本化周波数のまま録る
  await context.audioWorklet.addModule('recorder-worklet.js');
  const source = context.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(context, 'recorder-processor');
  const mute = context.createGain();
  mute.gain.value = 0; // 処理を駆動するため出力へつなぐが、音は出さない
  const rate = context.sampleRate;
  const state = {
    session, stream, context, source, node, rate,
    ring: new RingBuffer(Math.ceil(RING_SEC * rate)),
    scheduler: new InferenceScheduler(rate),
    busy: false,
  };
  history = [];
  nowSec = 0;
  currentEl.firstChild.textContent = '–.– ';
  node.port.onmessage = (event) => onChunk(state, event.data);
  source.connect(node).connect(mute).connect(context.destination);
  rec = state;
  recordButton.textContent = '録音停止';
  statusEl.textContent = '録音中（最初の表示は約 2 秒後）';
  draw();
}

async function stop() {
  const state = rec;
  rec = null;
  state.node.port.onmessage = null;
  state.source.disconnect();
  state.node.disconnect();
  state.stream.getTracks().forEach((t) => t.stop());
  await state.context.close();
  recordButton.textContent = '録音開始';
  statusEl.textContent = '停止（グラフは停止時点のまま）';
}

recordButton.addEventListener('click', async () => {
  recordButton.disabled = true;
  try {
    if (rec === null) await start();
    else await stop();
  } catch (error) {
    statusEl.textContent = `エラー: ${error}`;
    console.error(error);
  } finally {
    recordButton.disabled = false;
  }
});

// ---- グラフ ------------------------------------------------------------------------------

function draw() {
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const yMax = yAxisMax(history.map((p) => p.v));
  const now = Math.max(nowSec, SPAN_SEC); // 録音直後は左端を 0 秒に固定
  const sc = makeScale({ width, height, left: 36, right: 12, top: 10, bottom: 24, spanSec: SPAN_SEC, nowSec: now, yMax });

  ctx.font = '11px system-ui, sans-serif';
  ctx.fillStyle = '#666';
  ctx.strokeStyle = '#eee';
  ctx.lineWidth = 1;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let v = 0; v <= yMax; v += 2) {
    const y = sc.y(v);
    ctx.beginPath();
    ctx.moveTo(sc.left, y);
    ctx.lineTo(sc.right, y);
    ctx.stroke();
    ctx.fillText(String(v), sc.left - 4, y);
  }
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let ago = SPAN_SEC; ago >= 0; ago -= 5) {
    ctx.fillText(ago === 0 ? 'いま' : `-${ago}s`, sc.x(now - ago), sc.bottom + 6);
  }

  // 評価の話速帯の境界（早口の閾値ではない）
  ctx.strokeStyle = '#bbb';
  ctx.setLineDash([4, 4]);
  for (const b of BAND_BOUNDARIES) {
    if (b > yMax) continue;
    ctx.beginPath();
    ctx.moveTo(sc.left, sc.y(b));
    ctx.lineTo(sc.right, sc.y(b));
    ctx.stroke();
  }
  ctx.setLineDash([]);

  ctx.strokeStyle = '#999';
  ctx.strokeRect(sc.left, sc.top, sc.right - sc.left, sc.bottom - sc.top);

  if (history.length > 0) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(sc.left, sc.top, sc.right - sc.left, sc.bottom - sc.top);
    ctx.clip();
    ctx.strokeStyle = '#1565c0';
    ctx.lineWidth = 2;
    ctx.beginPath();
    history.forEach((p, i) => {
      const x = sc.x(p.t);
      const y = sc.y(p.v);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    const last = history[history.length - 1];
    ctx.fillStyle = '#1565c0';
    ctx.beginPath();
    ctx.arc(sc.x(last.t), sc.y(last.v), 3, 0, 2 * Math.PI);
    ctx.fill();
    ctx.restore();
  }
}

window.addEventListener('resize', draw);
draw();
