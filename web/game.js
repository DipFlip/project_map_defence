// Finger-gun Pong in the browser: a port of webcam_pong.py + pmd/scenes/hand_pong.py + pong.py.
//
// The webcam view fills the window (mirrored) and the court is drawn over it. Each player's
// bouncer is the line between their thumb tip and index tip; the leftmost hand is the left
// player, the rightmost the right player. Hand tracking is MediaPipe's hand landmarker,
// running in the page (WASM + WebGL), so nothing leaves the browser.
import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.1/vision_bundle.mjs";

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.1/wasm";
const MODEL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

// --- pong.py ---------------------------------------------------------------
const WIN_SCORE = 7;
const BALL_MM = 14;
const CRUISE_MM = 320; // mm/s the ball relaxes back to
const MIN_MM = 200, MAX_MM = 1500; // ball speed limits
const RELAX_S = 2.5; // time constant for a fast ball to slow back to cruising speed
const RESTITUTION = 0.85; // how much of the push along the bouncer's normal comes back
const GRIP = 0.35; // share of the bouncer's sideways motion (relative to the ball) passed to the ball
const MAX_PADDLE = 0.6; // bouncer length limit, fraction of the court's height
const PADDLE_HALF_THICK = 6;
// --- hand_pong.py ----------------------------------------------------------
const MIN_SIDEWAYS = 0.25; // after a hit the ball crosses the court at least this fast (x cruise), either way
const SEPARATION = 0.4; // after a hit the ball leaves the paddle at least this fast relative to it (x cruise)
const HIT_COOLDOWN_S = 0.3; // a paddle can't hit the ball again this soon (contact just pushes it clear)
const GRACE_S = 0.2; // keep a bouncer this long after its hand drops out (tracking blinks)
const PREDICT_S = 0.04; // extrapolate the bouncer towards "now" by at most this much
const JUMP = 0.3; // a hand appearing this far (x screen width) from the last one restarts its smoothing
// --- webcam_pong.py --------------------------------------------------------
const VIRTUAL_TABLE_MM = 1100; // the screen's width stands for this much table
const DIM_LEVELS = [0.35, 0.6, 1.0]; // video brightness under the game

const THUMB_TIP = 4, INDEX_TIP = 8;
const BONES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]];

// --- small 2D vector helpers -------------------------------------------------
const add = (a, b) => [a[0] + b[0], a[1] + b[1]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1]];
const mul = (a, k) => [a[0] * k, a[1] * k];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1];
const norm = (a) => Math.hypot(a[0], a[1]);
const perp = (v) => [-v[1], v[0]];
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const rgb = (c, a = 1) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
const rand = (lo, hi) => lo + Math.random() * (hi - lo);

// --- One Euro filter (pmd/filters.py), over a flat array ---------------------
function alpha(cutoff, dt) {
  const tau = 1 / (2 * Math.PI * cutoff);
  return 1 / (1 + tau / dt);
}

class OneEuro {
  constructor(minCutoff = 1.0, beta = 0.01, dCutoff = 2.0) {
    Object.assign(this, { minCutoff, beta, dCutoff, x: null, dx: null });
  }
  filter(x, dt) {
    if (this.x === null || dt <= 0) {
      this.x = x.slice();
      this.dx = x.map(() => 0);
      return this.x;
    }
    const ad = alpha(this.dCutoff, dt);
    this.dx = this.dx.map((d, i) => d + ad * ((x[i] - this.x[i]) / dt - d));
    const cutoff = this.minCutoff + this.beta * Math.hypot(...this.dx);
    const a = alpha(cutoff, dt);
    this.x = this.x.map((v, i) => v + a * (x[i] - v));
    return this.x;
  }
}

// --- synthesised sounds (pmd/sound.py) ---------------------------------------
class Sound {
  constructor() {
    this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    this.out = this.ctx.createGain();
    this.out.gain.value = 0.8;
    this.out.connect(this.ctx.destination);
    const n = { C5: 523.3, E5: 659.3, G5: 784.0, C6: 1046.5 };
    this.lib = {
      hit: [["square", 520, 0.08, 0.35, 0.7, 0], ["noise", 0, 0.05, 0.2, 0, 0]],
      wall: [["triangle", 300, 0.05, 0.3, 0, 0]],
      score: [["square", n.E5, 0.09, 0.3, 0, 0], ["square", n.C6, 0.2, 0.3, 0, 0.09]],
      win: [["square", n.C5, 0.12, 0.3, 0, 0], ["square", n.E5, 0.12, 0.3, 0, 0.12],
        ["square", n.G5, 0.12, 0.3, 0, 0.24], ["square", n.C6, 0.45, 0.3, 0, 0.36]],
      valve: [["triangle", 400, 0.08, 0.2, 1.3, 0]],
    };
  }
  play(name, vol = 1) {
    const ctx = this.ctx, t0 = ctx.currentTime;
    for (const [wave, freq, dur, v, slide, delay] of this.lib[name] || []) {
      const g = ctx.createGain(), t = t0 + delay;
      const level = v * vol * (wave === "square" ? 0.5 : 1);
      g.gain.setValueAtTime(0, t);
      g.gain.linearRampToValueAtTime(level, t + 0.005);
      g.gain.setValueAtTime(level, t + Math.max(0.006, dur - 0.08));
      g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
      g.connect(this.out);
      let src;
      if (wave === "noise") {
        const buf = ctx.createBuffer(1, Math.ceil(ctx.sampleRate * dur), ctx.sampleRate);
        const d = buf.getChannelData(0);
        let acc = 0;
        for (let i = 0; i < d.length; i++) d[i] = acc += 0.5 * (Math.random() * 2 - 1 - acc);
        src = ctx.createBufferSource();
        src.buffer = buf;
      } else {
        src = ctx.createOscillator();
        src.type = wave;
        src.frequency.setValueAtTime(freq, t);
        if (slide) src.frequency.linearRampToValueAtTime(freq * slide, t + dur);
      }
      src.connect(g);
      src.start(t);
      src.stop(t + dur + 0.02);
    }
  }
}

// --- particles (pmd/particles.py) --------------------------------------------
class Particles {
  constructor(max = 1500) {
    this.p = [];
    this.max = max;
  }
  burst(pos, color, { n = 20, speed = 200, life = 0.6, size = 4, spread = 2 * Math.PI, direction = 0, drag = 2, colors = null } = {}) {
    for (let i = 0; i < n; i++) {
      const ang = direction + (Math.random() - 0.5) * spread, sp = speed * (0.3 + 0.7 * Math.random());
      this._add(pos, [Math.cos(ang) * sp, Math.sin(ang) * sp], life * (0.6 + 0.4 * Math.random()),
        size * (0.6 + 0.8 * Math.random()), colors ? colors[Math.floor(Math.random() * colors.length)] : color, drag);
    }
  }
  ring(pos, color, radius, { n = 24, speed = 60, life = 0.4, size = 3 } = {}) {
    for (let i = 0; i < n; i++) {
      const d = [Math.cos(2 * Math.PI * i / n), Math.sin(2 * Math.PI * i / n)];
      this._add(add(pos, mul(d, radius)), mul(d, speed), life, size, color, 1);
    }
  }
  confetti(f, n = 150) {
    const colors = [[255, 80, 80], [255, 210, 60], [80, 230, 120], [240, 120, 255], [255, 255, 255], [255, 150, 40]];
    for (let i = 0; i < n; i++) {
      this._add([f.left + Math.random() * f.width, f.top + Math.random() * f.height * 0.2],
        [(Math.random() - 0.5) * 200, 150 + Math.random() * 250], 1.5 + Math.random() * 1.5, 3 + Math.random() * 4,
        colors[Math.floor(Math.random() * colors.length)], 0.3);
    }
  }
  _add(pos, v, life, size, color, drag) {
    this.p.push({ pos: pos.slice(), v, life, age: 0, size, color, drag });
    if (this.p.length > this.max) this.p.splice(0, this.p.length - this.max);
  }
  update(dt) {
    for (const q of this.p) {
      q.age += dt;
      q.v = mul(q.v, Math.exp(-q.drag * dt));
      q.pos = add(q.pos, mul(q.v, dt));
    }
    this.p = this.p.filter((q) => q.age < q.life);
  }
  draw(c) {
    for (const q of this.p) {
      const f = Math.min(1, (1 - q.age / q.life) * 1.5);
      c.fillStyle = rgb(q.color, f);
      c.beginPath();
      c.arc(q.pos[0], q.pos[1], Math.max(1, q.size * (0.4 + 0.6 * f)), 0, 2 * Math.PI);
      c.fill();
    }
  }
  clear() { this.p = []; }
}

// --- players (pmd/scenes/players.py) ------------------------------------------
const COLORS = [[70, 220, 120], [255, 150, 50]];
const NAMES = ["green", "orange"];

class Player {
  constructor(side, field, i) {
    this.side = side;
    this.color = COLORS[i];
    this.name = NAMES[i];
    this.endX = side === "left" ? field.left : field.right;
    this.dir = side === "left" ? 1 : -1; // +x points from this player towards the opponent
  }
}

class Hand {
  constructor() { this.reset(); }
  reset() {
    this.filter = new OneEuro(1.5, 0.004, 3.0);
    this.tips = null; // [thumb, index] in screen px, smoothed
    this.vel = [0, 0]; // bouncer velocity, px/s
    this.t = null; // capture time of the last frame used
    this.seen = -1e9; // capture time the hand was last seen
  }
}

// --- the game (Pong + HandPong) -----------------------------------------------
class HandPong {
  constructor(app) {
    this.app = app;
    this.field = app.field;
    this.mm = app.pxPerMm;
    this.r = BALL_MM * this.mm;
    this.thick = Math.max(6, Math.round(this.field.height * 0.012));
    this.players = [new Player("left", this.field, 0), new Player("right", this.field, 1)];
    this.hands = [new Hand(), new Hand()];
    this.frameN = -1;
    this.newMatch(app.now);
  }

  newMatch(now) {
    this.score = [0, 0];
    this.winner = null;
    this.winnerAt = 0;
    this.rally = 0;
    this.ball = null;
    this.vel = null;
    this.trail = [];
    this.serveTo = Math.random() < 0.5 ? 0 : 1;
    this.serveAt = now + 2.0;
    this.prevPaddles = {};
    this.flash = null;
    this.lastHit = [-1e9, -1e9];
  }

  // hands -> bouncers
  track() {
    const frame = this.app.frame;
    if (!frame || frame.n === this.frameN) return;
    this.frameN = frame.n;
    const tips = frame.hands.map((h) => [this.app.toScreen(h[THUMB_TIP]), this.app.toScreen(h[INDEX_TIP])]);
    const mx = (g) => (g[0][0] + g[1][0]) / 2;
    tips.sort((a, b) => mx(a) - mx(b));
    const bySide = {};
    if (tips.length >= 2) {
      bySide.left = tips[0];
      bySide.right = tips[tips.length - 1];
    } else if (tips.length) {
      bySide[mx(tips[0]) < this.field.centerX ? "left" : "right"] = tips[0];
    }
    const W = this.field.width;
    this.players.forEach((p, i) => {
      const h = this.hands[i], g = bySide[p.side];
      if (!g) {
        if (frame.t - h.seen > GRACE_S) h.reset();
        return;
      }
      if (h.tips && norm(sub(mul(add(g[0], g[1]), 0.5), mul(add(h.tips[0], h.tips[1]), 0.5))) > JUMP * W) h.reset();
      const dt = h.t !== null ? frame.t - h.t : 0;
      const x = h.filter.filter([...g[0], ...g[1]], dt);
      h.tips = [[x[0], x[1]], [x[2], x[3]]];
      const d = h.filter.dx;
      h.vel = [(d[0] + d[2]) / 2, (d[1] + d[3]) / 2];
      h.t = h.seen = frame.t;
    });
  }

  paddle(i) {
    const h = this.hands[i];
    if (!h.tips) return null;
    const lead = clamp(this.app.now - h.t, 0, PREDICT_S);
    const a = add(h.tips[0], mul(h.vel, lead)), b = add(h.tips[1], mul(h.vel, lead));
    const mid = mul(add(a, b), 0.5);
    let d = sub(b, a);
    const length = norm(d);
    const half = Math.min(length / 2, MAX_PADDLE * this.field.height / 2);
    d = mul(d, 1 / Math.max(length, 1e-6));
    return [sub(mid, mul(d, half)), add(mid, mul(d, half)), h.vel];
  }

  serve() {
    const p = this.players[this.serveTo];
    const ang = rand(-0.45, 0.45);
    const d = [-p.dir * Math.cos(ang), Math.sin(ang)]; // towards the player who conceded
    this.ball = [this.field.centerX, this.field.centerY];
    this.vel = mul(d, CRUISE_MM * this.mm * 0.9);
    this.rally = 0;
    this.trail = [];
    this.app.sound.play("valve");
    this.app.fx.ring(this.ball, [255, 255, 230], this.r * 1.5, { n: 18, speed: 120 });
  }

  bounce(i, normal, paddleVel, now) {
    if (now - this.lastHit[i] < HIT_COOLDOWN_S) {
      this.separate(normal, paddleVel);
      return;
    }
    // table Pong's moving-wall bounce, without forcing the ball towards the opponent
    const p = this.players[i];
    const rel = sub(this.vel, paddleVel);
    const vn = dot(rel, normal);
    if (vn >= 0) return; // ball was already moving away: not a hit
    const tangent = sub(rel, mul(normal, vn));
    let v = sub(sub(this.vel, mul(normal, (1 + RESTITUTION) * vn)), mul(tangent, GRIP));
    const speed = norm(v);
    this.vel = mul(v, clamp(speed, MIN_MM * this.mm, MAX_MM * this.mm) / Math.max(speed, 1e-6));
    this.rally += 1;
    this.flash = [i, now];
    const hard = Math.min(1, norm(this.vel) / (MAX_MM * this.mm * 0.6));
    this.app.sound.play("hit", 0.5 + 0.5 * hard);
    this.app.fx.burst(sub(this.ball, mul(normal, this.r)), p.color, {
      n: Math.round(14 + 30 * hard), speed: 150 + 400 * hard, life: 0.45, size: 4, spread: 2.2,
      direction: Math.atan2(normal[1], normal[0]), colors: [p.color, [255, 255, 255]],
    });
    this.lastHit[i] = now;
    const need = MIN_SIDEWAYS * CRUISE_MM * this.mm;
    if (Math.abs(this.vel[0]) < need) this.vel[0] = need * (Math.sign(this.vel[0]) || p.dir);
    this.separate(normal, paddleVel);
  }

  // make the ball move away from the paddle along the normal at least SEPARATION x cruise
  separate(normal, paddleVel) {
    const need = SEPARATION * CRUISE_MM * this.mm;
    const away = dot(sub(this.vel, paddleVel), normal);
    if (away < need) this.vel = add(this.vel, mul(normal, need - away));
    const speed = norm(this.vel), top = MAX_MM * this.mm;
    if (speed > top) this.vel = mul(this.vel, top / speed);
  }

  static closest(p, a, b) {
    const ab = sub(b, a);
    const t = clamp(dot(sub(p, a), ab) / Math.max(dot(ab, ab), 1e-6), 0, 1);
    return add(a, mul(ab, t));
  }

  static side(p, pad) {
    const [a, b] = pad;
    const t = mul(sub(b, a), 1 / Math.max(norm(sub(b, a)), 1e-6));
    return dot(sub(p, a), perp(t));
  }

  within(p, pad) {
    const [a, b] = pad, ab = sub(b, a);
    const L = Math.max(norm(ab), 1e-6);
    const u = dot(sub(p, a), ab) / L;
    return -this.r <= u && u <= L + this.r;
  }

  touch(i, pad, now) {
    const [a, b, u] = pad;
    const cp = HandPong.closest(this.ball, a, b);
    const d = sub(this.ball, cp);
    const dist = norm(d);
    if (dist >= this.r + PADDLE_HALF_THICK || dist < 1e-6) return false;
    const n = mul(d, 1 / dist);
    if (dot(sub(this.vel, u), n) >= 0) return false; // already moving away from the bouncer
    this.ball = add(cp, mul(n, this.r + PADDLE_HALF_THICK + 1));
    this.bounce(i, n, u, now);
    return true;
  }

  update(dt, now) {
    this.track();
    const paddles = [this.paddle(0), this.paddle(1)];
    if (this.ball === null && this.winner === null && paddles.some((p) => p === null)) {
      this.serveAt = Math.max(this.serveAt, now + 1.0); // serve only once both players are in
    }
    if (this.winner !== null) {
      if (now - this.winnerAt > 6) this.newMatch(now);
      return;
    }
    if (this.ball === null) {
      if (now >= this.serveAt) this.serve();
      this.prevPaddles = paddles;
      return;
    }
    const f = this.field;
    let speed = norm(this.vel);
    const cruise = CRUISE_MM * this.mm;
    if (speed > cruise) this.vel = mul(this.vel, (cruise + (speed - cruise) * Math.exp(-dt / RELAX_S)) / speed);
    const prev = this.ball.slice();
    const steps = Math.max(1, Math.ceil(norm(this.vel) * dt / (this.r * 0.5)));
    for (let s = 0; s < steps; s++) {
      this.ball = add(this.ball, mul(this.vel, dt / steps));
      if ((this.ball[1] < f.top + this.r && this.vel[1] < 0) || (this.ball[1] > f.bottom - this.r && this.vel[1] > 0)) {
        this.vel[1] = -this.vel[1];
        this.app.sound.play("wall", 0.6);
        this.app.fx.burst(this.ball, [200, 200, 200], { n: 6, speed: 120, life: 0.25, size: 2.5 });
      }
      let hit = false;
      for (let i = 0; i < 2 && !hit; i++) if (paddles[i]) hit = this.touch(i, paddles[i], now);
    }
    // fast swipes: the bouncer may have jumped across the ball between camera frames
    for (let i = 0; i < 2; i++) {
      const pad = paddles[i], old = this.prevPaddles[i];
      if (!pad || !old) continue;
      const s0 = HandPong.side(prev, old), s1 = HandPong.side(this.ball, pad);
      if (s0 * s1 < 0 && this.within(this.ball, pad)) {
        const [a, b, u] = pad;
        const n = mul(perp(mul(sub(b, a), 1 / norm(sub(b, a)))), Math.sign(s0));
        this.ball = add(HandPong.closest(this.ball, a, b), mul(n, this.r + PADDLE_HALF_THICK + 1));
        this.bounce(i, n, u, now);
      }
    }
    this.prevPaddles = paddles;
    this.trail.push(this.ball.slice());
    if (this.trail.length > 12) this.trail.shift();
    if (this.ball[0] < f.left - this.r || this.ball[0] > f.right + this.r) {
      const loser = this.ball[0] < f.left ? 0 : 1;
      const winner = 1 - loser;
      this.score[winner] += 1;
      const goalX = this.players[loser].endX, wc = this.players[winner].color;
      this.app.fx.burst([goalX, this.ball[1]], wc, {
        n: 60, speed: 500, life: 0.8, size: 5, direction: loser === 0 ? 0 : Math.PI, spread: 2.5,
        colors: [wc, [255, 255, 255]],
      });
      this.app.sound.play("score");
      this.ball = null;
      this.serveTo = loser;
      this.serveAt = now + 1.2;
      if (this.score[winner] >= WIN_SCORE) {
        this.winner = winner;
        this.winnerAt = now;
        this.app.sound.play("win");
        this.app.fx.confetti(f);
      }
    }
  }

  text(c, size, msg, [x, y], color) {
    c.font = `bold ${Math.round(size)}px system-ui, -apple-system, sans-serif`;
    c.textAlign = "center";
    c.textBaseline = "middle";
    c.fillStyle = "rgba(0,0,0,0.85)";
    c.fillText(msg, x + 3, y + 3);
    c.fillStyle = rgb(color);
    c.fillText(msg, x, y);
  }

  draw(c, now) {
    const f = this.field, big = f.height * 0.11, small = f.height * 0.05;
    c.lineCap = "butt";
    c.strokeStyle = "rgb(235,235,235)";
    c.lineWidth = 4;
    c.setLineDash([30, 30]);
    c.beginPath();
    c.moveTo(f.centerX, f.top);
    c.lineTo(f.centerX, f.bottom);
    c.stroke();
    c.setLineDash([]);
    for (const p of this.players) { // goal lines in the defender's colour
      c.strokeStyle = rgb(p.color);
      c.lineWidth = 8;
      c.beginPath();
      c.moveTo(p.endX, f.top);
      c.lineTo(p.endX, f.bottom);
      c.stroke();
    }
    this.players.forEach((p, i) => {
      this.text(c, big, String(this.score[i]), [f.centerX - p.dir * f.width * 0.08, f.top + f.height * 0.1], p.color);
    });

    const frame = this.app.frame;
    if (frame && this.app.debug) {
      c.strokeStyle = "white";
      c.lineWidth = 2;
      for (const h of frame.hands) {
        const pts = h.map((q) => this.app.toScreen(q));
        c.beginPath();
        for (const [a, b] of BONES) {
          c.moveTo(...pts[a]);
          c.lineTo(...pts[b]);
        }
        c.stroke();
      }
    }

    c.lineCap = "round";
    this.players.forEach((p, i) => {
      const pad = this.paddle(i);
      if (!pad) return;
      const [a, b] = pad;
      const lit = this.flash && this.flash[0] === i && now - this.flash[1] < 0.15;
      for (const [col, w] of [["black", this.thick * 2 + 6], [lit ? "white" : rgb(p.color), this.thick * 2]]) {
        c.strokeStyle = col;
        c.lineWidth = w;
        c.beginPath();
        c.moveTo(...a);
        c.lineTo(...b);
        c.stroke();
      }
    });

    if (this.ball) {
      const n = this.trail.length;
      this.trail.forEach((q, j) => {
        const k = (j + 1) / n;
        c.fillStyle = `rgb(${120 + 135 * k},${120 + 135 * k},${100 + 130 * k})`;
        c.beginPath();
        c.arc(q[0], q[1], this.r * (0.25 + 0.6 * k), 0, 2 * Math.PI);
        c.fill();
      });
      c.fillStyle = "black";
      c.beginPath();
      c.arc(this.ball[0], this.ball[1], this.r + 3, 0, 2 * Math.PI);
      c.fill();
      c.fillStyle = "rgb(255,255,230)";
      c.beginPath();
      c.arc(this.ball[0], this.ball[1], this.r, 0, 2 * Math.PI);
      c.fill();
    } else if (this.winner === null) {
      const waiting = this.players.filter((p, i) => !this.paddle(i));
      for (const p of waiting) {
        this.text(c, small, `${p.name}: show your hand`, [f.centerX - p.dir * f.width * 0.25, f.centerY], p.color);
      }
      if (!waiting.length) {
        const left = Math.max(0, this.serveAt - now);
        this.text(c, small, `serve in ${Math.round(left)}`, [f.centerX, f.centerY], [255, 255, 255]);
      }
    }
    if (this.winner !== null) {
      const w = this.players[this.winner];
      this.text(c, big, `${w.name} wins!`, [f.centerX, f.centerY], w.color);
    }
  }
}

// --- the page (webcam_pong.py's CamApp) -----------------------------------------
class CamApp {
  constructor(video, canvas, landmarker) {
    Object.assign(this, { video, canvas, landmarker });
    this.ctx = canvas.getContext("2d");
    this.sound = new Sound();
    this.fx = new Particles();
    this.frame = null;
    this.frameCount = 0;
    this.lastVideoTime = -1;
    this.debug = false;
    this.dim = 1;
    this.now = performance.now() / 1000;
    this.handHz = 0;
    this.drawFps = 0;
    this._hz = { t: this.now, n: 0, drawN: 0 };
    this.resize();
    addEventListener("resize", () => this.resize());
  }

  resize() {
    const W = innerWidth, H = innerHeight, dpr = devicePixelRatio || 1;
    this.canvas.width = Math.round(W * dpr);
    this.canvas.height = Math.round(H * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.W = W;
    this.H = H;
    this.pxPerMm = W / VIRTUAL_TABLE_MM;
    const m = 4;
    this.field = { left: m, top: m, right: W - m, bottom: H - m, width: W - 2 * m, height: H - 2 * m, centerX: W / 2, centerY: H / 2 };
    this.fx.clear();
    this.scene = new HandPong(this); // new size: new field
  }

  // fit the camera image to cover the window (cropping the overflow), keeping its proportions
  fit() {
    const fw = this.video.videoWidth, fh = this.video.videoHeight;
    this.scale = Math.max(this.W / fw, this.H / fh);
    this.fw = fw;
    this.fh = fh;
    this.offset = [(this.W - fw * this.scale) / 2, (this.H - fh * this.scale) / 2];
  }

  // normalised, mirrored camera coordinates (0..1) -> screen px, the same mapping the video is drawn with
  toScreen(q) {
    return [q[0] * this.fw * this.scale + this.offset[0], q[1] * this.fh * this.scale + this.offset[1]];
  }

  detect() {
    const v = this.video;
    if (v.readyState < 2 || v.currentTime === this.lastVideoTime) return;
    this.lastVideoTime = v.currentTime;
    const t = performance.now();
    const res = this.landmarker.detectForVideo(v, t);
    const hands = res.landmarks.map((lm) => lm.map((p) => [1 - p.x, p.y])); // mirrored like the video
    this.frame = { hands, t: t / 1000, n: ++this.frameCount };
    this._hz.n++;
  }

  step(dt, now) {
    this.now = now;
    const c = this.ctx;
    this.detect();
    this.fit();
    c.fillStyle = "black";
    c.fillRect(0, 0, this.W, this.H);
    if (this.video.readyState >= 2) {
      c.save();
      c.translate(this.W, 0);
      c.scale(-1, 1);
      c.filter = DIM_LEVELS[this.dim] < 1 ? `brightness(${DIM_LEVELS[this.dim]})` : "none";
      c.drawImage(this.video, this.W - this.offset[0] - this.fw * this.scale, this.offset[1], this.fw * this.scale, this.fh * this.scale);
      c.restore();
    }
    this.scene.update(dt, now);
    this.fx.update(dt);
    this.scene.draw(c, now);
    this.fx.draw(c);
    this._hz.drawN++;
    if (now - this._hz.t > 1) {
      this.handHz = this._hz.n / (now - this._hz.t);
      this.drawFps = this._hz.drawN / (now - this._hz.t);
      this._hz = { t: now, n: 0, drawN: 0 };
    }
    c.font = "14px system-ui, sans-serif";
    c.textAlign = "left";
    c.textBaseline = "bottom";
    c.fillStyle = "rgba(200,200,200,0.8)";
    c.fillText(`hands ${this.handHz.toFixed(0)} Hz   draw ${this.drawFps.toFixed(0)} fps`, 12, this.H - 8);
  }

  key(e) {
    const k = e.key.toLowerCase();
    if (k === "r") {
      this.fx.clear();
      this.scene.newMatch(this.now);
    } else if (k === "v") {
      this.dim = (this.dim + 1) % DIM_LEVELS.length;
    } else if (k === "d") {
      this.debug = !this.debug;
    } else if (k === "f") {
      if (document.fullscreenElement) document.exitFullscreen();
      else document.documentElement.requestFullscreen?.();
    }
  }
}

async function start() {
  const go = document.getElementById("go"), err = document.getElementById("err");
  go.disabled = true;
  err.textContent = "";
  try {
    go.textContent = "Starting camera…";
    const video = document.getElementById("video");
    video.srcObject = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } }, audio: false,
    });
    await video.play();
    go.textContent = "Loading hand model…";
    const vision = await FilesetResolver.forVisionTasks(WASM);
    const opts = (delegate) => ({
      baseOptions: { modelAssetPath: MODEL, delegate }, runningMode: "VIDEO", numHands: 4,
      minHandDetectionConfidence: 0.5, minHandPresenceConfidence: 0.5, minTrackingConfidence: 0.5,
    });
    let landmarker;
    try {
      landmarker = await HandLandmarker.createFromOptions(vision, opts("GPU"));
    } catch {
      landmarker = await HandLandmarker.createFromOptions(vision, opts("CPU"));
    }
    document.getElementById("start").remove();
    const app = new CamApp(video, document.getElementById("canvas"), landmarker);
    addEventListener("keydown", (e) => app.key(e));
    let prev = performance.now() / 1000;
    const loop = () => {
      const now = performance.now() / 1000;
      app.step(Math.min(now - prev, 0.1), now);
      prev = now;
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  } catch (e) {
    console.error(e);
    err.textContent = e.name === "NotAllowedError" ? "Camera access was denied." : `Could not start: ${e.message || e}`;
    go.disabled = false;
    go.textContent = "Try again";
  }
}

document.getElementById("go").addEventListener("click", start);
