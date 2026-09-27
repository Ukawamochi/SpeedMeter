// web/dsp.js の照合テスト（node:test、npm の依存なし）。
// 実行: node --test web/tests/   （pytest の tests/test_web_demo.py からも呼ぶ）
//
// 対数メルは tests/fixtures/melspec/ の参照値と、tests/test_melspec_fixtures.py と同じ
// 許容誤差で照合する（全要素 2e-3、参照値が -5 より大きい要素 1e-4。自然対数の単位の絶対誤差）。

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import {
  dft,
  hannWindow,
  logMelSpectrogram,
  melFilterbank,
  mixToMono,
  numFrames,
  parseWholeOutput,
  parseWindowOutputs,
  resampleSinc,
  windowStarts,
} from '../dsp.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const FIXTURE_DIR = join(ROOT, 'tests', 'fixtures', 'melspec');

const ATOL_ALL = 2e-3;
const ATOL_ENERGY = 1e-4;
const ENERGY_THRESHOLD = -5.0;
const SIGNAL_NAMES = ['multitone_with_silence', 'sine_1000hz'];

function load(name) {
  return JSON.parse(readFileSync(join(FIXTURE_DIR, `${name}.json`), 'utf-8'));
}

// 参照値の input_conversion: float32(waveform_int16) / 32768
export function int16ToFloat32(values) {
  return Float32Array.from(values, (v) => Math.fround(v / 32768));
}

for (const name of SIGNAL_NAMES) {
  test(`対数メルが参照値と一致する: ${name}`, () => {
    const payload = load(name);
    const { numFrames: frames, nMels, data } = logMelSpectrogram(int16ToFloat32(payload.waveform_int16));
    assert.equal(frames, payload.num_frames);
    assert.equal(nMels, 80);
    assert.equal(payload.log_mel.length, frames);
    let maxAll = 0;
    let maxEnergy = 0;
    let energyCount = 0;
    for (let t = 0; t < frames; t++) {
      for (let m = 0; m < 80; m++) {
        const ref = payload.log_mel[t][m];
        const diff = Math.abs(data[t * 80 + m] - ref);
        maxAll = Math.max(maxAll, diff);
        if (ref > ENERGY_THRESHOLD) {
          energyCount++;
          maxEnergy = Math.max(maxEnergy, diff);
        }
      }
    }
    console.log(`# ${name}: max|diff| all=${maxAll.toExponential(3)} energy(>-5)=${maxEnergy.toExponential(3)} (n=${energyCount})`);
    assert.ok(energyCount > 0);
    assert.ok(maxAll <= ATOL_ALL, `全要素の最大誤差 ${maxAll} > ${ATOL_ALL}`);
    assert.ok(maxEnergy <= ATOL_ENERGY, `値の大きい帯の最大誤差 ${maxEnergy} > ${ATOL_ENERGY}`);
  });
}

test('無音だけのフレームは log(1e-6) になる', () => {
  const { data } = logMelSpectrogram(int16ToFloat32(load('multitone_with_silence').waveform_int16));
  const expected = Math.fround(Math.log(Math.fround(1e-6)));
  for (let i = 0; i < 9 * 80; i++) assert.ok(Math.abs(data[i] - expected) < 1e-6);
});

// フィルタバンクの許容誤差は 2e-6 とする（tests/test_melspec_fixtures.py は 1e-6）。torch の
// float32 の冪（10 ** x）が82点の端点のうち1点（38番目）で倍精度からの丸めと1ulp異なり、その
// 点を頂点・端にもつフィルタで最大 1.3e-6 の差になるため。対数メルの照合は同じ許容誤差で行う。
const ATOL_FILTERBANK = 2e-6;

test('メルフィルタバンクが参照値と一致する', () => {
  const payload = JSON.parse(readFileSync(join(FIXTURE_DIR, 'mel_filterbank.json'), 'utf-8'));
  assert.equal(payload.num_bins, 201);
  assert.equal(payload.n_mels, 80);
  const fb = melFilterbank();
  const rebuilt = new Float64Array(201 * 80);
  payload.bands.forEach((band, mel) => {
    band.weights.forEach((w, k) => {
      rebuilt[(band.start_bin + k) * 80 + mel] = w;
    });
  });
  let maxDiff = 0;
  for (let i = 0; i < rebuilt.length; i++) maxDiff = Math.max(maxDiff, Math.abs(rebuilt[i] - fb[i]));
  console.log(`# mel_filterbank: max|diff|=${maxDiff.toExponential(3)}`);
  assert.ok(maxDiff <= ATOL_FILTERBANK);
});

test('DFT が定義どおりの総和と一致する（400点）', () => {
  const n = 400;
  const x = Array.from({ length: n }, (_, i) => Math.sin(i * 0.37) + 0.3 * Math.cos(i * 1.91) + (i % 7) * 0.01);
  const [re, im] = dft(x);
  for (const k of [0, 1, 7, 50, 199, 200, 201, 399]) {
    let r = 0;
    let s = 0;
    for (let i = 0; i < n; i++) {
      r += x[i] * Math.cos((2 * Math.PI * k * i) / n);
      s -= x[i] * Math.sin((2 * Math.PI * k * i) / n);
    }
    assert.ok(Math.abs(re[k] - r) < 1e-9 && Math.abs(im[k] - s) < 1e-9, `k=${k}`);
  }
});

test('hann 窓は periodic', () => {
  const w = hannWindow(400);
  assert.equal(w[0], 0);
  assert.ok(Math.abs(w[200] - 1) < 1e-15);
  assert.ok(Math.abs(w[1] - w[399]) < 1e-15);
});

test('フレーム数は 1 + 標本数 // 160（2.0秒窓で201）', () => {
  assert.equal(numFrames(32000), 201);
  assert.equal(logMelSpectrogram(new Float32Array(32000)).numFrames, 201);
  assert.equal(numFrames(48000), 301);
});

test('窓は2.0秒・0.25秒ずらし・末尾の端数を捨てる', () => {
  assert.deepEqual(windowStarts(31999), []);
  assert.deepEqual(windowStarts(32000), [0]);
  assert.deepEqual(windowStarts(48000), [0, 4000, 8000, 12000, 16000]);
  assert.deepEqual(windowStarts(47999), [0, 4000, 8000, 12000]);
});

test('出力のパース: 毎秒モーラ数 = モーラ数 ÷ 2.0秒（窓）、÷ 録音の秒数（全体）', () => {
  const parsed = parseWindowOutputs(Float32Array.of(10, 3), [0, 4000]);
  assert.equal(parsed[0].moraPerSec, 5);
  assert.equal(parsed[1].startSec, 0.25);
  assert.equal(parsed[1].endSec, 2.25);
  const whole = parseWholeOutput(12, 48000);
  assert.equal(whole.durationSec, 3);
  assert.equal(whole.moraPerSec, 4);
});

test('モノラル化はチャンネル平均', () => {
  assert.deepEqual(Array.from(mixToMono([Float32Array.of(1, 0), Float32Array.of(0, 1)])), [0.5, 0.5]);
});

for (const fromRate of [48000, 44100]) {
  test(`再標本化 ${fromRate}Hz → 16kHz: 1kHz の正弦波が保たれる`, () => {
    const seconds = 1;
    const x = Float32Array.from({ length: fromRate * seconds }, (_, i) => 0.5 * Math.sin((2 * Math.PI * 1000 * i) / fromRate));
    const y = resampleSinc(x, fromRate, 16000);
    assert.equal(y.length, 16000 * seconds);
    let maxErr = 0;
    for (let j = 100; j < y.length - 100; j++) {
      maxErr = Math.max(maxErr, Math.abs(y[j] - 0.5 * Math.sin((2 * Math.PI * 1000 * j) / 16000)));
    }
    console.log(`# resample ${fromRate}: max|err| (端の100標本を除く)=${maxErr.toExponential(3)}`);
    assert.ok(maxErr < 5e-3);
  });

  test(`再標本化 ${fromRate}Hz → 16kHz: 7.9kHz を超える成分は落ちる`, () => {
    const x = Float32Array.from({ length: fromRate }, (_, i) => 0.5 * Math.sin((2 * Math.PI * 12000 * i) / fromRate));
    const y = resampleSinc(x, fromRate, 16000);
    let rms = 0;
    for (let j = 200; j < y.length - 200; j++) rms += y[j] * y[j];
    rms = Math.sqrt(rms / (y.length - 400));
    assert.ok(rms < 0.01, `rms=${rms}`);
  });
}
