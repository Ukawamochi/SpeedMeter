// web/realtime.js のテスト（node:test、マイクを使わない）。実行: node --test web/tests/realtime.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';

import { resampleSinc } from '../dsp.js';
import {
  BAND_BOUNDARIES,
  InferenceScheduler,
  RingBuffer,
  extractWindow16k,
  makeScale,
  nativeWindowLength,
  pruneHistory,
  yAxisMax,
} from '../realtime.js';

const ramp = (from, n) => Float32Array.from({ length: n }, (_, i) => from + i);

test('環形バッファ: 絶対位置で読み、折り返しをまたいでも順序が保たれる', () => {
  const ring = new RingBuffer(10);
  ring.push(ramp(0, 7));
  assert.deepEqual(Array.from(ring.read(7, 3)), [4, 5, 6]);
  ring.push(ramp(7, 6)); // 0..12 を書いた。容量10なので 3..12 が残る
  assert.equal(ring.totalWritten, 13);
  assert.deepEqual(Array.from(ring.read(13, 10)), Array.from(ramp(3, 10)));
  assert.deepEqual(Array.from(ring.read(9, 4)), [5, 6, 7, 8]);
  assert.equal(ring.read(13, 11), null); // 上書き済み
  assert.equal(ring.read(14, 2), null); // 未来
});

test('環形バッファ: 容量より長い塊は末尾だけが残る', () => {
  const ring = new RingBuffer(4);
  ring.push(ramp(0, 3));
  ring.push(ramp(3, 9)); // 0..11
  assert.equal(ring.totalWritten, 12);
  assert.deepEqual(Array.from(ring.read(12, 4)), [8, 9, 10, 11]);
});

test('窓の切り出し: 16kHz の 32000 標本になり、中身はまとめて変換したものと一致する', () => {
  const rate = 48000;
  const total = rate * 3;
  const signal = Float32Array.from({ length: total }, (_, i) => 0.3 * Math.sin((2 * Math.PI * 440 * i) / rate));
  const ring = new RingBuffer(rate * 4);
  for (let i = 0; i < total; i += 128) ring.push(signal.subarray(i, Math.min(total, i + 128)));
  const end = rate * 3;
  const win = extractWindow16k(ring, end, rate);
  assert.equal(win.length, 32000);
  // 窓の終わり（いま）= 16kHz の 48000 標本目。最後の数十標本を除き、全体の変換と一致する。
  const whole = resampleSinc(signal, rate, 16000);
  let maxDiff = 0;
  for (let j = 0; j < 32000 - 50; j++) maxDiff = Math.max(maxDiff, Math.abs(win[j] - whole[48000 - 32000 + j]));
  assert.ok(maxDiff < 1e-5, `maxDiff=${maxDiff}`);
  assert.equal(extractWindow16k(ring, nativeWindowLength(rate) - 1, rate), null);
});

test('スケジュール: 0.25 秒ごとに1回、最初は 2.0 秒 + 余白がたまってから', () => {
  const rate = 16000;
  const s = new InferenceScheduler(rate);
  const first = nativeWindowLength(rate);
  assert.equal(first, 32160);
  assert.equal(s.update(first - 1, false), null);
  assert.equal(s.update(first, false), first);
  assert.equal(s.update(first + 3999, false), null); // 次の時点の前
  assert.equal(s.update(first + 4000, false), first + 4000);
  assert.equal(s.runs, 2);
  assert.equal(s.skipped, 0);
});

test('スケジュール: 推論中に来た時点は間引き、遅れを溜めず最新の時点だけを実行する', () => {
  const rate = 16000;
  const s = new InferenceScheduler(rate, 32000);
  assert.equal(s.update(32000, false), 32000); // 時点0
  assert.equal(s.update(36000, true), null); // 時点1: 推論中 → 間引き
  assert.equal(s.update(40000, true), null); // 時点2: 推論中 → 間引き
  assert.equal(s.skipped, 2);
  assert.equal(s.update(40100, false), null); // 同じ時点は再実行しない
  assert.equal(s.update(52000, false), 52000); // 時点3・4を飛ばして時点5だけを実行
  assert.equal(s.skipped, 4);
  assert.equal(s.runs, 2);
});

test('グラフの座標: 右端がいま、左端が spanSec 秒前、下端が0、上端が yMax', () => {
  const sc = makeScale({ width: 600, height: 300, left: 40, right: 10, top: 10, bottom: 30, spanSec: 30, nowSec: 100, yMax: 10 });
  assert.equal(sc.x(100), 590);
  assert.equal(sc.x(70), 40);
  assert.equal(sc.x(85), 315);
  assert.equal(sc.y(0), 270);
  assert.equal(sc.y(10), 10);
  assert.equal(sc.y(5), 140);
});

test('縦軸の上限は既定 10、超えたら 2 刻みで切り上げる', () => {
  assert.equal(yAxisMax([]), 10);
  assert.equal(yAxisMax([3, 9.9]), 10);
  assert.equal(yAxisMax([10.1]), 12);
  assert.equal(yAxisMax([NaN, 13]), 14);
});

test('履歴の刈り込み: 範囲外の直前の1点だけを残す', () => {
  const pts = [0, 1, 2, 3, 4, 5].map((t) => ({ t, v: t }));
  assert.deepEqual(pruneHistory(pts, 5, 2.5).map((p) => p.t), [2, 3, 4, 5]);
  assert.deepEqual(pruneHistory(pts, 5, 10).map((p) => p.t), [0, 1, 2, 3, 4, 5]);
  assert.deepEqual(pruneHistory([], 5, 10), []);
});

test('補助線は評価の話速帯の境界 4・6・8', () => {
  assert.deepEqual(BAND_BOUNDARIES, [4, 6, 8]);
});
