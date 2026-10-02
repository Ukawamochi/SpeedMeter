// 発表の残り時間タイマー（DOM に依存しない）。時刻は Date.now() から数える（タブが裏でも狂わない）。
export class CountdownTimer {
  constructor(durationMs, now = () => Date.now()) {
    this.durationMs = durationMs;
    this.now = now;
    this.elapsedMs = 0;     // 停止中までに経過した時間
    this.startedAt = null;  // 動いている間の開始時刻
  }

  get running() { return this.startedAt !== null; }

  start() { if (!this.running) this.startedAt = this.now(); }

  pause() {
    if (!this.running) return;
    this.elapsedMs += this.now() - this.startedAt;
    this.startedAt = null;
  }

  toggle() { if (this.running) this.pause(); else this.start(); }

  // 止める。経過を 0 に戻す（持ち時間はそのまま）
  reset() { this.startedAt = null; this.elapsedMs = 0; }

  // 持ち時間を決め直す。経過も 0 に戻す
  setDuration(ms) { this.durationMs = ms; this.reset(); }

  // 残り時間（ミリ秒）。持ち時間を超えたら負になる
  remainingMs() {
    const elapsed = this.elapsedMs + (this.running ? this.now() - this.startedAt : 0);
    return this.durationMs - elapsed;
  }
}

// 残り時間の表示。超過は "+m:ss"。端数は切り上げ（残り 0.4 秒は 0:01）
export function formatRemaining(ms) {
  const over = ms < 0;
  const sec = over ? Math.floor(-ms / 1000) : Math.ceil(ms / 1000);
  const text = `${Math.floor(sec / 60)}:${String(sec % 60).padStart(2, '0')}`;
  return over && sec > 0 ? `+${text}` : text;
}

// 'normal' | 'warn'（残り 1 分以内） | 'over'（持ち時間を超過）
export function timerLevel(ms) {
  if (ms < 0) return 'over';
  if (ms <= 60_000) return 'warn';
  return 'normal';
}

// 入力 "m" または "m:ss" をミリ秒にする。読めない・0 以下・180 分超は null
export function parseDuration(text) {
  const m = /^\s*(\d{1,3})(?::([0-5]?\d))?\s*$/.exec(text);
  if (!m) return null;
  const sec = Number(m[1]) * 60 + Number(m[2] ?? 0);
  return sec > 0 && sec <= 180 * 60 ? sec * 1000 : null;
}
