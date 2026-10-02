// 管理ページ。録音・推論はこのページで動かし、最前面の小窓（Document Picture-in-Picture）にはバーだけを描く。
// 小窓は開いたページと同じ JavaScript の文脈で動くので、推論の結果を直接 DOM に書ける。
// 窓の「最前面」は Chrome が保証する（他のアプリを前面にしても残る）。開くにはユーザーの操作（ボタン）が要る。
import { ort } from './lib/ort.js';
import { WINDOW_SEC, logMelSpectrogram } from './lib/dsp.js';
import { RingBuffer, InferenceScheduler, extractWindow16k } from './lib/realtime.js';
import { PaceAverager } from './lib/pace.js';
import { CountdownTimer, formatRemaining, timerLevel, parseDuration } from './lib/timer.js';

const RING_SEC = 4;       // 環形バッファの長さ（窓 2.0 秒 + 余裕）
const PACE_MAX = 12;      // 話速バーの右端（字/秒）
const AVERAGE_SEC = 15;   // 平均を取る長さ（秒）
const SILENCE_CHARS = 2;  // 窓（2.0 秒）の字数がこれを下回ったら無音とみなす（仮の値。あとで決める）
// preferInitialWindowPlacement: Chrome が覚えている前回の大きさでなく、指定の大きさで開く
const PIP_SIZE = { width: 400, height: 120, preferInitialWindowPlacement: true };
const DEFAULT_DURATION_MS = 10 * 60_000;
const SETTINGS_KEY = 'speedmeter.settings';

const startButton = document.getElementById('start');
const statusEl = document.getElementById('status');

function setStatus(text) { statusEl.textContent = text; }

function describe(error) {
  return error?.name && error.name !== 'Error' ? `${error.name}: ${error.message}` : String(error?.message ?? error);
}

// どの段階で失敗したかを、エラーの文面に足す
async function at(stage, promise) {
  try { return await promise; } catch (error) { throw new Error(`${stage}: ${describe(error)}`); }
}

// ---- 設定（持ち時間・モデル）。このブラウザに保存する ---------------------------------------
function loadSettings() {
  try { return JSON.parse(localStorage.getItem(SETTINGS_KEY)) ?? {}; } catch { return {}; }
}
function saveSettings(settings) {
  try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings)); } catch { /* 保存できなくても動く */ }
}

// ---- 小窓の中身 --------------------------------------------------------------------------
function buildBar(doc, models) {
  const link = doc.createElement('link');
  link.rel = 'stylesheet';
  link.href = new URL('bar.css', location.href).href;
  doc.head.append(link);
  doc.title = 'SpeedMeter';

  doc.body.innerHTML = `
    <div class="pip" id="main">
      <div class="pip-top">
        <div class="paces">
          <span class="pace-value" id="pace-value" title="現在">–.–</span>
          <span class="pace-value avg" id="avg-value" title="過去 15 秒の平均（無音を除く）">–.–</span>
          <span class="pace-unit">字/秒</span>
        </div>
        <div class="timer" id="timer">
          <span class="timer-text" id="timer-text">0:00</span>
          <button id="toggle" type="button" aria-label="開始・停止">▶</button>
          <button id="reset" type="button" aria-label="リセット">↺</button>
          <button id="open-settings" type="button" aria-label="設定">⚙</button>
        </div>
      </div>
      <div class="pace-bar" role="img" aria-label="現在の話速"><div class="pace-fill" id="pace-fill"></div><i></i><i></i></div>
      <div class="pace-bar avg" role="img" aria-label="過去 15 秒の平均"><div class="pace-fill" id="avg-fill"></div><i></i><i></i></div>
      <div class="time-bar" role="img" aria-label="残り時間"><div class="time-fill" id="time-fill"></div></div>
    </div>
    <form class="pip settings" id="settings" hidden>
      <label>時間<input id="duration" type="text" inputmode="numeric" placeholder="10 または 10:30" autocomplete="off"></label>
      <div class="row">
        <label>モデル<select id="model">${models.map((m) => `<option value="${m.id}">${m.id}</option>`).join('')}</select></label>
        <button type="submit" aria-label="決定">✓</button>
      </div>
    </form>`;
  const $ = (id) => doc.getElementById(id);
  return {
    doc, main: $('main'), settings: $('settings'),
    paceValue: $('pace-value'), paceFill: $('pace-fill'), avgValue: $('avg-value'), avgFill: $('avg-fill'),
    timer: $('timer'), timerText: $('timer-text'), toggle: $('toggle'), reset: $('reset'), openSettings: $('open-settings'),
    timeFill: $('time-fill'), duration: $('duration'), model: $('model'),
  };
}

function setPace(valueEl, fillEl, perSec) {
  valueEl.textContent = perSec === null ? '–.–' : perSec.toFixed(1);
  fillEl.style.width = perSec === null ? '0' : `${Math.min(Math.max(perSec / PACE_MAX, 0), 1) * 100}%`;
}

// result: { chars: 窓の字数, perSec: 現在の話速 }。平均は無音を除いた過去 AVERAGE_SEC 秒
function showPace(view, averager, result) {
  setPace(view.paceValue, view.paceFill, result.perSec);
  setPace(view.avgValue, view.avgFill, averager.average());
}

function showTimer(view, timer) {
  const remaining = timer.remainingMs();
  const level = timerLevel(remaining);
  const cls = level === 'normal' ? '' : level;
  view.timerText.textContent = formatRemaining(remaining);
  view.timerText.className = `timer-text ${cls}`.trim();
  view.timer.classList.toggle('paused', !timer.running);
  view.toggle.textContent = timer.running ? '❚❚' : '▶';
  view.timeFill.className = `time-fill ${cls}`.trim();
  view.timeFill.style.width = `${Math.min(Math.max(remaining / timer.durationMs, 0), 1) * 100}%`;
}

// タイマーの操作と設定パネル。音は鳴らさない。onModel(id) はモデルの切り替えを頼む
function bindControls(view, timer, settings, onModel) {
  const refresh = () => showTimer(view, timer);
  view.toggle.addEventListener('click', () => { timer.toggle(); refresh(); });
  view.reset.addEventListener('click', () => { timer.reset(); refresh(); });

  const setPanel = (open) => {
    view.main.hidden = open;
    view.settings.hidden = !open;
    if (open) {
      view.duration.value = formatRemaining(timer.durationMs);
      view.duration.setCustomValidity('');
      view.model.value = settings.model;
      view.duration.focus();
      view.duration.select();
    }
  };
  view.openSettings.addEventListener('click', () => setPanel(true));
  view.duration.addEventListener('input', () => view.duration.setCustomValidity(''));
  view.settings.addEventListener('submit', (event) => {
    event.preventDefault();
    const ms = parseDuration(view.duration.value);
    if (ms === null) {
      view.duration.setCustomValidity('1〜180 分で、10 または 10:30 の形で入力');
      view.duration.reportValidity();
      return;
    }
    if (ms !== timer.durationMs) timer.setDuration(ms);
    settings.durationMs = ms;
    if (view.model.value !== settings.model) { settings.model = view.model.value; onModel(settings.model); }
    saveSettings(settings);
    setPanel(false);
    refresh();
  });
  view.settings.addEventListener('keydown', (event) => { if (event.key === 'Escape') setPanel(false); });
}

// ---- 計測の開始と停止 -----------------------------------------------------------------------
async function loadModels() {
  const response = await at('models/models.json を読めない（scripts/build_extension.py を実行して拡張機能を更新する）', fetch('models/models.json'));
  if (!response.ok) throw new Error('models/models.json が無い（scripts/build_extension.py を実行する）');
  return response.json();
}

function createSession(models, id) {
  const entry = models.find((m) => m.id === id) ?? models[0];
  return at(`モデル ${entry.file} の読み込み`, ort.InferenceSession.create(`models/${entry.file}`, { executionProviders: ['wasm'] }));
}

let current = null; // 実行中の計測

async function begin() {
  if (!('documentPictureInPicture' in window)) {
    throw new Error('Document Picture-in-Picture に未対応（Chrome 116 以降が必要）');
  }
  const models = await loadModels();
  const settings = { durationMs: DEFAULT_DURATION_MS, model: models[0].id, ...loadSettings() };
  if (!models.some((m) => m.id === settings.model)) settings.model = models[0].id;

  // 小窓はユーザー操作の直後にしか開けないので、時間のかかる準備（モデル・マイク）より先に開く
  const pipWindow = await at('小窓を開く', window.documentPictureInPicture.requestWindow(PIP_SIZE));
  const view = buildBar(pipWindow.document, models);
  const timer = new CountdownTimer(settings.durationMs);
  let onnx = null;
  let modelToken = 0; // 切り替えの途中で次の切り替えが来たとき、古い方を捨てる
  const switchModel = async (id) => {
    const token = ++modelToken;
    const next = await createSession(models, id);
    if (token === modelToken) onnx = next;
  };
  bindControls(view, timer, settings, (id) => switchModel(id).catch((e) => setStatus(describe(e))));
  showTimer(view, timer);
  const repaint = setInterval(() => showTimer(view, timer), 250); // 音声が来ない間（マイクの許可待ちなど）も表示を進める
  pipWindow.addEventListener('pagehide', () => clearInterval(repaint));

  let stream; let context;
  try {
    await switchModel(settings.model);
    stream = await at('マイク', navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    }));
    context = new AudioContext(); // 機器の標本化周波数のまま録り、推論の直前に 16kHz へ変換する
    await at('録音の準備', context.audioWorklet.addModule('lib/recorder-worklet.js'));
    const source = context.createMediaStreamSource(stream);
    const node = new AudioWorkletNode(context, 'recorder-processor');
    const mute = context.createGain();
    mute.gain.value = 0; // 処理を駆動するため出力へつなぐが、音は出さない
    const rate = context.sampleRate;
    const ring = new RingBuffer(Math.ceil(RING_SEC * rate));
    const scheduler = new InferenceScheduler(rate);
    const averager = new PaceAverager({ spanSec: AVERAGE_SEC, silenceChars: SILENCE_CHARS, windowSec: WINDOW_SEC });
    let busy = false;
    let lastPaint = 0;

    // 画面の更新は音声のブロック到着に合わせる（タブが裏でもタイマーの更新が止まらない）
    node.port.onmessage = (event) => {
      ring.push(event.data);
      const now = performance.now();
      if (now - lastPaint > 250) { lastPaint = now; showTimer(view, timer); }
      const at = scheduler.update(ring.totalWritten, busy);
      if (at === null) return;
      busy = true;
      infer(onnx, ring, at, rate)
        .then((chars) => {
          if (chars === null) return;
          averager.push(at / rate, chars);
          showPace(view, averager, { chars, perSec: chars / WINDOW_SEC });
        })
        .catch((error) => setStatus(describe(error)))
        .finally(() => { busy = false; });
    };
    source.connect(node).connect(mute).connect(context.destination);

    current = { stream, context, node, source };
    startButton.disabled = true;
    setStatus('');
    pipWindow.addEventListener('pagehide', () => finish());
  } catch (error) {
    clearInterval(repaint);
    pipWindow.close();
    stream?.getTracks().forEach((t) => t.stop());
    await context?.close();
    throw error;
  }
}

async function infer(onnx, ring, at, rate) {
  const window16k = extractWindow16k(ring, at, rate);
  if (window16k === null) return null;
  const mel = logMelSpectrogram(window16k);
  const input = new ort.Tensor('float32', mel.data, [1, mel.numFrames, mel.nMels]);
  const outputs = await onnx.run({ log_mel: input });
  return outputs.mora.data[0]; // 窓の字数（モデルの出力するモーラ数をそのまま字数として扱う）
}

async function finish() {
  const s = current;
  if (!s) return;
  current = null;
  s.node.port.onmessage = null;
  s.source.disconnect();
  s.node.disconnect();
  s.stream.getTracks().forEach((t) => t.stop());
  await s.context.close();
  startButton.disabled = false;
}

startButton.addEventListener('click', async () => {
  startButton.disabled = true;
  setStatus('');
  try {
    await begin();
  } catch (error) {
    console.error(describe(error));
    setStatus(describe(error));
    startButton.disabled = false;
  }
});
