// 過去 spanSec 秒の平均の話速。無音の時点は平均の対象から外す（= 無音の秒数を分母から引く）。
// 推論は 0.25 秒ごとなので、各時点は同じ長さの区間を表す。平均 = 無音でない時点の話速の平均。
// chars は 1 つの窓（windowSec 秒）に含まれる字数の推定値。chars が silenceChars を下回る窓は無音とみなす。
export class PaceAverager {
  constructor({ spanSec = 15, silenceChars = 2, windowSec = 2.0 } = {}) {
    this.spanSec = spanSec;
    this.silenceChars = silenceChars;
    this.windowSec = windowSec;
    this.points = []; // { t, rate }。rate は毎秒の字数。無音は null
  }

  // t: 窓の終わりの時刻（秒）
  push(t, chars) {
    this.points.push({ t, rate: chars < this.silenceChars ? null : chars / this.windowSec });
    const from = t - this.spanSec;
    while (this.points.length > 0 && this.points[0].t <= from) this.points.shift();
  }

  // 無音を除いた平均（字/秒）。有音の時点が無ければ null
  average() {
    let sum = 0;
    let n = 0;
    for (const p of this.points) if (p.rate !== null) { sum += p.rate; n += 1; }
    return n === 0 ? null : sum / n;
  }
}
