// 確認用ページの録音（AudioWorklet）。入力の各ブロック（128 標本）をチャンネル平均で
// モノラルにして、そのままの標本化周波数（AudioContext.sampleRate）で主スレッドへ送る。
// 16kHz への再標本化は主スレッド（web/dsp.js の resampleSinc）で録音の停止後にまとめて行う。
class RecorderProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (input && input.length > 0) {
      const n = input[0].length;
      const mono = new Float32Array(n);
      for (const channel of input) for (let i = 0; i < n; i++) mono[i] += channel[i];
      if (input.length > 1) for (let i = 0; i < n; i++) mono[i] /= input.length;
      this.port.postMessage(mono, [mono.buffer]);
    }
    return true;
  }
}

registerProcessor('recorder-processor', RecorderProcessor);
