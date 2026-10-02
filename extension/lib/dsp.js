// 確認用ページの信号処理（docs/spec.md「特徴量」・冒頭の窓の定義を JS に移したもの）。
//
// ブラウザ（web/app.js）と Node の照合テスト（web/tests/）の両方から ES module として読む。
// DOM や Web Audio には依存しない純粋な関数だけを置く。
//
// 対数メルスペクトログラムの計算は src/spkrate/features/melspec.py（torchaudio の
// MelSpectrogram の既定）と同じ定義である:
//   - periodic な hann 窓 w[n] = 0.5 - 0.5 cos(2πn/400)、win_length = n_fft = 400
//   - center=True: 両端を 200 標本ずつ reflect（端の標本を含めない鏡映）で詰め、160 標本ずつ
//     ずらして 400 標本を切り出す。フレーム数は 1 + 標本数 // 160
//   - 片側の DFT（201 ビン）、正規化なし、パワー |X|^2
//   - HTK メル尺度、0〜8000Hz を 82 点に等分した三角フィルタ 80 本、norm=None
//   - 自然対数 log(mel + 1e-6)
// 途中の計算は JS の number（倍精度）で行い、出力は Float32Array に丸める（ONNX の入力は
// float32）。torch の float32 計算との差は tests/test_melspec_fixtures.py と同じ許容誤差で
// web/tests/melspec.test.mjs が照合する。

export const MEL_CONFIG = Object.freeze({
  sampleRate: 16000,
  nFft: 400,
  hopLength: 160,
  winLength: 400,
  nMels: 80,
  fMin: 0.0,
  fMax: 8000.0,
  power: 2.0,
  logOffset: 1e-6,
});

// docs/spec.md 冒頭: モデルの入力単位は固定長窓 2.0 秒、推論間隔 0.25 秒。
export const WINDOW_SEC = 2.0;
export const HOP_SEC = 0.25;

const N_FFT = MEL_CONFIG.nFft;
const N_BINS = N_FFT / 2 + 1; // 201
const PAD = N_FFT / 2; // 200

export function numFrames(numSamples, hopLength = MEL_CONFIG.hopLength) {
  return 1 + Math.floor(numSamples / hopLength);
}

// periodic な hann 窓（torch.hann_window(400) の既定）。
export function hannWindow(n = N_FFT) {
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n);
  return w;
}

function hzToMelHtk(f) {
  return 2595.0 * Math.log10(1.0 + f / 700.0);
}


// torch.linspace（float32）と同じ並べ方（端点も float32 に丸める）: 前半は start + i*step、後半は end - (steps-1-i)*step。
function linspaceF32(start, end, steps) {
  const f = Math.fround;
  start = f(start);
  end = f(end);
  const out = new Float64Array(steps);
  const step = f((end - start) / (steps - 1));
  const half = Math.floor(steps / 2);
  for (let i = 0; i < steps; i++) {
    out[i] = i < half ? f(start + f(i * step)) : f(end - f((steps - 1 - i) * step));
  }
  return out;
}

// メルフィルタバンク。torchaudio.functional.melscale_fbanks（mel_scale="htk", norm=None）と
// 同じ式を、torch と同じく float32 の丸め（Math.fround）で計算する。
// 戻り値は (201 ビン, 80) を行優先で並べた Float64Array（fb[bin * 80 + mel]）。
export function melFilterbank(cfg = MEL_CONFIG) {
  const f = Math.fround;
  const nMels = cfg.nMels;
  const nyquist = Math.floor(cfg.sampleRate / 2);
  const allFreqs = linspaceF32(0, nyquist, N_BINS);
  const mPts = linspaceF32(hzToMelHtk(cfg.fMin), hzToMelHtk(cfg.fMax), nMels + 2);
  const fPts = mPts.map((m) => f(f(700.0 * f(f(Math.pow(10.0, f(m / 2595.0))) - 1.0))));
  const fb = new Float64Array(N_BINS * nMels);
  for (let k = 0; k < N_BINS; k++) {
    for (let m = 0; m < nMels; m++) {
      const down = f(f(-f(fPts[m] - allFreqs[k])) / f(fPts[m + 1] - fPts[m]));
      const up = f(f(fPts[m + 2] - allFreqs[k]) / f(fPts[m + 2] - fPts[m + 1]));
      fb[k * nMels + m] = Math.max(0, Math.min(down, up));
    }
  }
  return fb;
}

// ---- 400 点の複素 DFT（混合基数の Cooley-Tukey。400 = 2^4 * 5^2） ----------------------

function smallestFactor(n) {
  for (let p = 2; p * p <= n; p++) if (n % p === 0) return p;
  return n;
}

// 長さ n の複素列（re, im）の DFT を返す。twCos/twSin は長さ N（最上位の長さ）の回転因子表、
// scale = N / n。
function dftRecursive(re, im, n, twCos, twSin, scale) {
  if (n === 1) return [Float64Array.of(re[0]), Float64Array.of(im[0])];
  const p = smallestFactor(n);
  const m = n / p;
  const subs = [];
  for (let r = 0; r < p; r++) {
    const sRe = new Float64Array(m);
    const sIm = new Float64Array(m);
    for (let k = 0; k < m; k++) {
      sRe[k] = re[k * p + r];
      sIm[k] = im[k * p + r];
    }
    subs.push(m === 1 ? [sRe, sIm] : dftRecursive(sRe, sIm, m, twCos, twSin, scale * p));
  }
  const outRe = new Float64Array(n);
  const outIm = new Float64Array(n);
  const bigN = twCos.length;
  for (let k = 0; k < n; k++) {
    const km = k % m;
    let accRe = 0;
    let accIm = 0;
    for (let r = 0; r < p; r++) {
      // W_n^{r k} = exp(-2πi r k / n)。表は N 点なので指数を scale 倍する。
      const idx = ((r * k) % n) * scale % bigN;
      const c = twCos[idx];
      const s = twSin[idx];
      const yRe = subs[r][0][km];
      const yIm = subs[r][1][km];
      accRe += yRe * c + yIm * s;
      accIm += yIm * c - yRe * s;
    }
    outRe[k] = accRe;
    outIm[k] = accIm;
  }
  return [outRe, outIm];
}

// 実数列 x（長さ N）の DFT。X[k] = Σ x[n] exp(-2πi k n / N)。検査用にも公開する。
export function dft(x) {
  const n = x.length;
  const twCos = new Float64Array(n);
  const twSin = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    twCos[i] = Math.cos((2 * Math.PI * i) / n);
    twSin[i] = Math.sin((2 * Math.PI * i) / n);
  }
  return dftRecursive(Float64Array.from(x), new Float64Array(n), n, twCos, twSin, 1);
}

// ---- 対数メルスペクトログラム -----------------------------------------------------------

let cached = null;

function tables() {
  if (cached === null) {
    const twCos = new Float64Array(N_FFT);
    const twSin = new Float64Array(N_FFT);
    for (let i = 0; i < N_FFT; i++) {
      twCos[i] = Math.cos((2 * Math.PI * i) / N_FFT);
      twSin[i] = Math.sin((2 * Math.PI * i) / N_FFT);
    }
    cached = { window: hannWindow(N_FFT), fb: melFilterbank(), twCos, twSin };
  }
  return cached;
}

// reflect の詰め物（torch.nn.functional.pad(mode="reflect")。端の標本を含めない鏡映）。
function reflectPad(samples, pad) {
  const n = samples.length;
  if (n <= pad) throw new Error(`reflect の詰め物には ${pad + 1} 標本以上が必要: ${n}`);
  const out = new Float64Array(n + 2 * pad);
  for (let i = 0; i < pad; i++) out[i] = samples[pad - i];
  for (let i = 0; i < n; i++) out[pad + i] = samples[i];
  for (let i = 0; i < pad; i++) out[pad + n + i] = samples[n - 2 - i];
  return out;
}

// 16kHz モノラルの波形（Float32Array など）から対数メル (フレーム数, 80) を計算する。
// 戻り値の data は行優先（data[frame * 80 + mel]）で、ONNX の入力 log_mel の1件ぶんと同じ並び。
export function logMelSpectrogram(samples) {
  const { window, fb, twCos, twSin } = tables();
  const nMels = MEL_CONFIG.nMels;
  const hop = MEL_CONFIG.hopLength;
  const padded = reflectPad(samples, PAD);
  const frames = numFrames(samples.length, hop);
  const data = new Float32Array(frames * nMels);
  const re = new Float64Array(N_FFT);
  const im = new Float64Array(N_FFT);
  const power = new Float64Array(N_BINS);
  for (let t = 0; t < frames; t++) {
    const offset = t * hop;
    for (let i = 0; i < N_FFT; i++) {
      re[i] = padded[offset + i] * window[i];
      im[i] = 0;
    }
    const [xRe, xIm] = dftRecursive(re, im, N_FFT, twCos, twSin, 1);
    for (let k = 0; k < N_BINS; k++) power[k] = xRe[k] * xRe[k] + xIm[k] * xIm[k];
    for (let m = 0; m < nMels; m++) {
      let acc = 0;
      for (let k = 0; k < N_BINS; k++) {
        const w = fb[k * nMels + m];
        if (w !== 0) acc += power[k] * w;
      }
      data[t * nMels + m] = Math.log(acc + MEL_CONFIG.logOffset);
    }
  }
  return { numFrames: frames, nMels, data };
}

// ---- 窓の切り出し（docs/spec.md「窓単位の評価」dev_window の窓の規約と同じ） -------------
// 長さ 2.0 秒の窓を 0.25 秒ずつずらす。窓全体が音声に収まるものだけを取り、詰め物をしない
// （末尾の端数は捨てる）。2.0 秒未満の音声は窓を持たない。
export function windowStarts(numSamples, sampleRate = MEL_CONFIG.sampleRate, windowSec = WINDOW_SEC, hopSec = HOP_SEC) {
  const win = Math.round(windowSec * sampleRate);
  const hop = Math.round(hopSec * sampleRate);
  const starts = [];
  for (let s = 0; s + win <= numSamples; s += hop) starts.push(s);
  return starts;
}

// ---- 再標本化（確認用ページで使う方法。Python 側の librosa/soxr とは別の実装） ---------
// 帯域制限の sinc 補間（hann 窓付き）。torchaudio.functional.resample の既定
// （resampling_method="sinc_interp_hann"、lowpass_filter_width=6、rolloff=0.99）と同じ核を、
// 出力標本ごとに直接評価する。波形の外側は 0 とみなす。出力長は ceil(N * to / from)。
export function resampleSinc(samples, fromRate, toRate, { lowpassFilterWidth = 6, rolloff = 0.99 } = {}) {
  if (fromRate === toRate) return Float32Array.from(samples);
  const n = samples.length;
  const outLen = Math.ceil((n * toRate) / fromRate);
  const out = new Float32Array(outLen);
  const baseFreq = Math.min(fromRate, toRate) * rolloff;
  const width = Math.ceil((lowpassFilterWidth * fromRate) / baseFreq);
  const gain = baseFreq / fromRate;
  for (let j = 0; j < outLen; j++) {
    const center = (j * fromRate) / toRate; // 入力の標本位置
    const lo = Math.max(0, Math.floor(center) - width);
    const hi = Math.min(n - 1, Math.ceil(center) + width);
    let acc = 0;
    for (let i = lo; i <= hi; i++) {
      const t = ((i - center) / fromRate) * baseFreq;
      if (t < -lowpassFilterWidth || t > lowpassFilterWidth) continue;
      const win = Math.cos((t * Math.PI) / lowpassFilterWidth / 2) ** 2;
      const sinc = t === 0 ? 1 : Math.sin(Math.PI * t) / (Math.PI * t);
      acc += samples[i] * sinc * win;
    }
    out[j] = acc * gain;
  }
  return out;
}

// 多チャンネル（Float32Array の配列）をチャンネル平均でモノラルにする
// （spkrate.eval.audio.to_mono と同じ、チャンネル方向の平均）。
export function mixToMono(channels) {
  if (channels.length === 1) return Float32Array.from(channels[0]);
  const n = channels[0].length;
  const out = new Float32Array(n);
  for (const ch of channels) for (let i = 0; i < n; i++) out[i] += ch[i];
  for (let i = 0; i < n; i++) out[i] /= channels.length;
  return out;
}

// ---- 出力のパース ------------------------------------------------------------------
// ONNX の出力 mora は (batch,) で、各要素が入力区間のモーラ数（毎秒モーラ数ではない）。
// docs/spec.md 冒頭「表示する話速: 出力を窓長で割った毎秒モーラ数」。
export function parseWindowOutputs(mora, starts, sampleRate = MEL_CONFIG.sampleRate, windowSec = WINDOW_SEC) {
  return Array.from(mora, (value, i) => ({
    startSec: starts[i] / sampleRate,
    endSec: starts[i] / sampleRate + windowSec,
    mora: value,
    moraPerSec: value / windowSec,
  }));
}

// 確認用: クリップ全体を1回入力したときの出力。毎秒モーラ数 = モーラ数 ÷ 録音の秒数。
export function parseWholeOutput(mora, numSamples, sampleRate = MEL_CONFIG.sampleRate) {
  const durationSec = numSamples / sampleRate;
  return { durationSec, mora, moraPerSec: mora / durationSec };
}

// ---- ONNX の入力の組み立て（ページと照合テストで共通） --------------------------------
// 入力 log_mel は (batch, frames, 80) の float32。正規化はグラフの中なので、ここでは行わない。

// 窓ごと（docs/spec.md の窓）: 各窓（32000 標本）の対数メルを並べて (窓数, 201, 80)。
// 対数メルは窓ごとに切り出した波形から計算する（窓の両端で reflect の詰め物をする）。
export function buildWindowInput(samples) {
  const win = Math.round(WINDOW_SEC * MEL_CONFIG.sampleRate);
  const starts = windowStarts(samples.length);
  const frames = numFrames(win);
  const nMels = MEL_CONFIG.nMels;
  const data = new Float32Array(starts.length * frames * nMels);
  starts.forEach((s, i) => {
    data.set(logMelSpectrogram(samples.subarray(s, s + win)).data, i * frames * nMels);
  });
  return { starts, dims: [starts.length, frames, nMels], data };
}

// 確認用: クリップ全体を1件として (1, 1 + 標本数 // 160, 80)。
export function buildWholeInput(samples) {
  const { numFrames: frames, nMels, data } = logMelSpectrogram(samples);
  return { dims: [1, frames, nMels], data };
}
