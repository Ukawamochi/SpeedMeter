// 確認用ページのリアルタイム表示の部品（DOM・Web Audio に依存しない純粋な処理）。
// ブラウザ（web/app.js）と Node の照合テスト（web/tests/realtime.test.mjs）から読む。
//
// docs/spec.md 冒頭: 入力単位は 2.0 秒窓、推論間隔 0.25 秒、表示する話速は出力 ÷ 窓長。
// 録音中は「直近 2.0 秒」の窓で 0.25 秒ごとに推論する。時刻はすべて録音の標本数から数える
// （壁時計ではない）。

import { WINDOW_SEC, HOP_SEC, MEL_CONFIG, resampleSinc } from './dsp.js';

// ---- 環形バッファ（ネイティブの標本化周波数の直近の音声） -------------------------------
// 絶対位置（録音開始からの標本番号）で読み出せる。容量を超えた古い標本は上書きされる。
export class RingBuffer {
  constructor(capacity) {
    this.capacity = capacity;
    this.buffer = new Float32Array(capacity);
    this.totalWritten = 0; // これまでに書いた標本数（= 次に書く標本の絶対位置）
  }

  push(chunk) {
    const cap = this.capacity;
    let src = chunk;
    if (src.length > cap) src = src.subarray(src.length - cap); // 容量を超える分は古い側を捨てる
    const skipped = chunk.length - src.length;
    const start = (this.totalWritten + skipped) % cap;
    const first = Math.min(src.length, cap - start);
    this.buffer.set(src.subarray(0, first), start);
    if (first < src.length) this.buffer.set(src.subarray(first), 0);
    this.totalWritten += chunk.length;
  }

  // 絶対位置 [end - length, end) の標本を返す。まだ書かれていない・既に上書きされた範囲なら null。
  read(end, length) {
    const begin = end - length;
    if (begin < 0 || end > this.totalWritten || begin < this.totalWritten - this.capacity) return null;
    const out = new Float32Array(length);
    const cap = this.capacity;
    const start = begin % cap;
    const first = Math.min(length, cap - start);
    out.set(this.buffer.subarray(start, start + first), 0);
    if (first < length) out.set(this.buffer.subarray(0, length - first), first);
    return out;
  }
}

// ---- 窓の切り出しと 16kHz への変換 --------------------------------------------------
// 16kHz の窓（32000 標本）を作るのに要るネイティブの標本数。sinc 補間の左端の影響を避けるため、
// 窓の前に余白（margin）を足して変換し、変換後の末尾 32000 標本を使う。窓の終わり（いま）より
// 先の標本は無いので、末尾の数標本は補間の核が片側だけになる（0 とみなす。resampleSinc の規約）。
export const RESAMPLE_MARGIN_SEC = 0.01;

export function nativeWindowLength(nativeRate, windowSec = WINDOW_SEC, marginSec = RESAMPLE_MARGIN_SEC) {
  return Math.ceil(windowSec * nativeRate) + Math.ceil(marginSec * nativeRate);
}

// ネイティブの標本の絶対位置 end で終わる、16kHz・2.0 秒（32000 標本）の窓を返す。足りなければ null。
export function extractWindow16k(ring, end, nativeRate) {
  const target = Math.round(WINDOW_SEC * MEL_CONFIG.sampleRate);
  const native = ring.read(end, nativeWindowLength(nativeRate));
  if (native === null) return null;
  const converted = resampleSinc(native, nativeRate, MEL_CONFIG.sampleRate);
  return converted.subarray(converted.length - target);
}

// ---- 推論のスケジュール（0.25 秒ごと。間に合わなければ間引く） -------------------------
// 推論の時点 k（k = 0, 1, …）は、録音の標本数が first + k * hop に達した時点（窓の終わり）。
// first は最初の窓に要る標本数（既定は 2.0 秒 + 変換の余白 = nativeWindowLength）。
// 音声が届くたびに update(totalWritten, busy) を呼ぶ。最新の時点が未処理なら、
//   - 推論中（busy）: その時点は実行しない（間引き）
//   - 空き: 最新の時点だけを実行する。その間に過ぎた古い時点は間引く（遅れを溜めない）
// 戻り値は実行する時点の窓の終わり（ネイティブの標本の絶対位置）か null。
export class InferenceScheduler {
  constructor(nativeRate, firstSamples = nativeWindowLength(nativeRate), hopSec = HOP_SEC) {
    this.firstSamples = firstSamples;
    this.hopSamples = Math.round(hopSec * nativeRate);
    this.lastTick = -1;
    this.runs = 0;
    this.skipped = 0;
  }

  latestTick(totalWritten) {
    if (totalWritten < this.firstSamples) return -1;
    return Math.floor((totalWritten - this.firstSamples) / this.hopSamples);
  }

  update(totalWritten, busy) {
    const tick = this.latestTick(totalWritten);
    if (tick <= this.lastTick) return null;
    if (busy) {
      this.skipped += tick - this.lastTick;
      this.lastTick = tick;
      return null;
    }
    this.skipped += tick - this.lastTick - 1;
    this.lastTick = tick;
    this.runs += 1;
    return this.firstSamples + tick * this.hopSamples;
  }
}

// ---- グラフ ------------------------------------------------------------------------
// 評価で使う話速帯の境界（docs/spec.md「評価指標」の 4・6・8 モーラ/秒）。早口の閾値ではない
// （閾値は仕様に未定義。docs/questions.md）。
export const BAND_BOUNDARIES = [4, 6, 8];

// 横軸: 直近 spanSec 秒（右端がいま）。縦軸: 0〜yMax モーラ/秒（上が大きい）。
export function makeScale({ width, height, left, right, top, bottom, spanSec, nowSec, yMax }) {
  const plotW = width - left - right;
  const plotH = height - top - bottom;
  return {
    x: (t) => left + ((t - (nowSec - spanSec)) / spanSec) * plotW,
    y: (v) => top + (1 - v / yMax) * plotH,
    left,
    right: width - right,
    top,
    bottom: height - bottom,
  };
}

// 縦軸の上限: 既定 10 モーラ/秒。表示範囲の値がそれを超えたら 2 刻みで切り上げる。
export function yAxisMax(values, minMax = 10) {
  let max = minMax;
  for (const v of values) if (Number.isFinite(v) && v > max) max = v;
  return Math.ceil(max / 2) * 2;
}

// 表示範囲（直近 spanSec 秒）より古い点を捨てる。線を左端までつなぐため、範囲外の直前の1点は残す。
export function pruneHistory(points, nowSec, spanSec) {
  const from = nowSec - spanSec;
  let firstInside = points.findIndex((p) => p.t >= from);
  if (firstInside === -1) firstInside = points.length;
  return points.slice(Math.max(0, firstInside - 1));
}
