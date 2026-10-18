// Replay saved restaurant trajectories and animate their decisions, rewards and comparisons.
(() => {
const RAW = JSON.parse(document.getElementById("traces").textContent);
const DATA = RAW.traces, TSEED = RAW.meta;
const ACTIONS = [[1,2],[1,4],[2,2],[2,4],[3,4],[4,4],[4,6],[5,6],[6,6],[7,8],[8,8]];
const actionId = (size, cap) => ACTIONS.findIndex(([a, b]) => a === size && b === cap);
const NS = "http://www.w3.org/2000/svg";
const svg = document.getElementById("scene");
const COMPAT = {1:[2,4],2:[2,4],3:[4],4:[4,6],5:[6],6:[6],7:[8],8:[8]};
const XS = [560, 730, 900, 1070];
const TPOS = {1:[XS[0],170],2:[XS[1],170],3:[XS[2],170],4:[XS[3],170],
              5:[XS[0],365],6:[XS[1],365],7:[XS[2],365],8:[XS[3],365],
              9:[XS[0],570],10:[XS[1],570],11:[XS[2],570],12:[XS[3],570]};
const RAD = {2:22,4:28,6:34,8:40};
const ENTRY = [-40, 100], EXIT_L = [-60, 700], EXIT_R = [1260, 700];
const QX = 70, QY0 = 196, QDY = 41, BY = 100, BX0 = 64, BDX = 78;
const COL = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const C = {btn: COL("--btn"), guest: COL("--guest"), hl: COL("--hl"), free: COL("--free"), lost: COL("--lost"),
           line: COL("--line"), ink: COL("--ink"), muted: COL("--muted"), floor: COL("--floor")};
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
const SCEN = {off_peak:"Off-Peak", normal:"Normal", peak:"Peak", super_peak:"Super Peak"};
const RULE = {
  "Best-fit": "Hand-written rule: pick the seating that wastes the fewest seats; ties go to the longest-waiting party.",
  "FIFO": "Hand-written rule: serve the longest-waiting party that has a free table it fits.",
  "DQN A": "Trained DQN, default 64×64 network. It takes the move with the highest predicted value.",
  "DQN C": "Trained DQN, one 64-unit layer. It takes the move with the highest predicted value.",
  "PPO A": "Trained PPO, default 64×64 network. Line thickness shows how likely it was to pick each table.",
  "PPO C": "Trained PPO, one 64-unit layer. Line thickness shows how likely it was to pick each table.",
  "PPO G": "Trained PPO with party-to-table attention. Line thickness shows how likely it was to pick each table."
};
const HOSTS = Object.keys(RULE);

let policy = "PPO G", scen = "super_peak", speed = 1, playing = true, stepOnce = false, gen = 0;
let parties, tables, served, lost, ret, layers, trayCount = 0, card;
const fmt = v => (v < 0 ? "−" : v > 0 ? "+" : "") + Math.abs(v).toFixed(2);
const fx = document.getElementById("fx");
const fW = document.getElementById("fW"), fL = document.getElementById("fL"), fR = document.getElementById("fR");
const center = e => { const r = e.getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2]; };
function pop(e, cls) { e.classList.remove(cls); void e.offsetWidth; e.classList.add(cls); }
// Show the waiting/loss terms for the current recorded interval.
function setFormula(w, l, show) {
  fW.textContent = w; fL.textContent = l;
  fL.parentElement.classList.toggle("hot", l > 0);
  fR.textContent = show ? fmt(-0.1 * w - 1.2 * l) : "…";
}
/* Every person who waited (amber) and every person lost (red) flies up into the formula,
   the result is computed, and the result then drops into Return. */
async function rewardTokens(tk, waitEls, lostEls) {
  const w = tk.end.waitedPersons, l = tk.end.lostPersons, r0 = ret, r1 = tk.totals.return;
  setFormula(0, 0, false);
  const tW = center(fW), tL = center(fL);
  const mk = (src, color, target) => {
    const [x, y] = center(src);
    const d = document.createElement("i"); d.className = "tok"; d.style.background = color;
    d.style.transform = `translate(${x}px,${y}px)`; fx.appendChild(d);
    return {d, x, y, target};
  };
  const toks = [...waitEls.slice(0, w).map(e => mk(e, C.guest, "w")), ...lostEls.slice(0, l).map(e => mk(e, C.lost, "l"))];
  const nW = toks.filter(t => t.target === "w").length, nL = toks.length - nW;
  let cw = 0, cl = 0;
  const flyDur = toks.length ? 1100 : 300;
  await anim(flyDur, (k, raw) => {
    toks.forEach((t, i) => {
      const u = Math.max(0, Math.min(1, raw * 1.6 - (i / Math.max(1, toks.length)) * 0.6));
      const e = ease(u), [tx, ty] = t.target === "w" ? tW : tL;
      const x = lerp(t.x, tx, e), y = lerp(t.y, ty, e) - Math.sin(e * Math.PI) * 60;
      t.d.style.transform = `translate(${x}px,${y}px) scale(${lerp(1, .55, e)})`;
      t.d.style.opacity = u >= 1 ? 0 : 1;
      if (u >= 1 && !t.done) {
        t.done = true;
        if (t.target === "w") { cw++; fW.textContent = Math.round(cw / nW * w); pop(fW, "bump"); }
        else { cl++; fL.textContent = Math.round(cl / nL * l); pop(fL, "bump"); }
      }
    });
  });
  toks.forEach(t => t.d.remove());
  setFormula(w, l, true); pop(fR, "bump");
  // the result travels into Return, which then counts to its new value
  const ghost = document.createElement("span"); ghost.className = "ghost" + (l > 0 ? " hot" : "");
  ghost.textContent = fR.textContent; fx.appendChild(ghost);
  const [gx, gy] = center(fR), sRet = document.getElementById("sRet"), [rx, ry] = center(sRet);
  await anim(700, k => {
    ghost.style.transform = `translate(${lerp(gx, rx, k)}px,${lerp(gy, ry, k) - Math.sin(k * Math.PI) * 26}px) translate(-50%,-50%)`;
    ghost.style.opacity = k < .85 ? 1 : (1 - k) / .15;
  });
  ghost.remove();
  pop(sRet.parentElement, l > 0 ? "hit" : "tick");
  await anim(450, k => { sRet.textContent = lerp(r0, r1, k).toFixed(1); });
  ret = r1;
}
const TRAY_X = 44, TRAY_Y = 700;

// Create an SVG element with attributes in the correct namespace.
function el(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
const ease = t => t < .5 ? 4*t*t*t : 1 - Math.pow(-2*t + 2, 3) / 2;
const ABORT = Symbol("abort");
// Advance an animation only while playing; a new replay generation cancels stale work.
function anim(dur, fn) {
  const g = gen;
  dur = reduced ? dur * 0.25 : dur;
  return new Promise((res, rej) => {
    let t = 0, last = null;
    function f(now) {
      if (g !== gen) { rej(ABORT); return; }
      if (last === null) last = now;
      const dt = Math.min(now - last, 50); last = now;
      if (playing) t += dt * speed;
      const k = Math.min(1, t / dur);
      fn(ease(k), k);
      if (k < 1) requestAnimationFrame(f); else res();
    }
    requestAnimationFrame(f);
  });
}
const pause = ms => anim(ms, () => {});
// Wait for playback to resume, unless another replay has replaced this one.
function untilPlaying() {
  const g = gen;
  return new Promise((res, rej) => {
    (function f() { if (g !== gen) return rej(ABORT); playing ? res() : requestAnimationFrame(f); })();
  });
}
const lerp = (a, b, t) => a + (b - a) * t;
// Interpolate the guest and loss colours for gradual visual transitions.
function mixColor(a, b, t) {
  const p = h => [1,3,5].map(i => parseInt(h.slice(i, i+2), 16));
  const A = p(a), B = p(b);
  return "rgb(" + A.map((v, i) => Math.round(lerp(v, B[i], t))).join(",") + ")";
}
const clockText = tick => { const m = tick * 5; return String(Math.floor(m/60)).padStart(2,"0") + ":" + String(m%60).padStart(2,"0"); };

// Static scene
// Rebuild the dining room, queue guides and table graphics for a fresh replay.
function buildScene() {
  svg.innerHTML = "";
  const defs = el("defs", {}, svg);
  const flt = el("filter", {id:"glow", x:"-50%", y:"-50%", width:"200%", height:"200%"}, defs);
  el("feGaussianBlur", {stdDeviation:"5", result:"b"}, flt);
  const mg = el("feMerge", {}, flt); el("feMergeNode", {in:"b"}, mg); el("feMergeNode", {in:"SourceGraphic"}, mg);
  const bg = el("g", {}, svg);
  el("rect", {x:0, y:0, width:1200, height:740, fill:C.floor}, bg);
  el("line", {x1:470, y1:40, x2:470, y2:712, stroke:C.line, "stroke-width":1}, bg);
  el("text", {x:40, y:62, class:"lbl"}, bg).textContent = "Just arrived";
  el("text", {x:40, y:160, class:"lbl"}, bg).textContent = "Waiting queue";
  el("text", {x:510, y:62, class:"lbl"}, bg).textContent = "Dining room";
  el("text", {x:40, y:682, class:"lbl", fill:C.lost}, bg).textContent = "Lost tonight";
  const lg = el("g", {transform:"translate(520,712)"}, bg);
  const item = (x, draw, text) => { draw(x); el("text", {x:x + 18, y:5, class:"legend"}, lg).textContent = text; };
  item(0, x => el("circle", {cx:x + 5, cy:0, r:5, fill:C.guest}, lg), "one guest");
  item(108, x => el("circle", {cx:x + 6, cy:0, r:6, fill:"none", stroke:C.hl, "stroke-width":2}, lg), "dining time left");
  item(262, x => el("line", {x1:x - 2, y1:0, x2:x + 13, y2:0, stroke:C.muted, "stroke-width":2, "stroke-linecap":"round"}, lg), "options (PPO: thicker = more likely)");
  item(530, x => el("rect", {x:x - 1, y:-6, width:14, height:12, rx:4, fill:"none", stroke:C.lost, "stroke-width":1.5}, lg), "passed over");
  // door marker
  el("path", {d:"M 8 82 L 8 118", stroke:C.muted, "stroke-width":3, "stroke-linecap":"round"}, bg);
  // queue slot guides
  for (let i = 0; i < 12; i++) el("line", {x1:40, y1:QY0+i*QDY+17, x2:440, y2:QY0+i*QDY+17, stroke:C.line, "stroke-width":1, opacity:.5}, bg);
  layers = {
    lines: el("g", {}, svg), tables: el("g", {}, svg),
    parties: el("g", {}, svg), fly: el("g", {}, svg)
  };
  tables = new Map();
  for (let id = 1; id <= 12; id++) {
    const cap = id <= 4 ? 2 : id <= 8 ? 4 : id <= 11 ? 6 : 8;
    const [x, y] = TPOS[id], R = RAD[cap];
    const g = el("g", {transform:`translate(${x},${y})`}, layers.tables);
    const ringR = R + 24, circ = 2 * Math.PI * ringR;
    el("circle", {r:ringR, fill:"none", stroke:C.line, "stroke-width":3}, g);
    const ring = el("circle", {r:ringR, fill:"none", stroke:C.hl, "stroke-width":3.5, "stroke-linecap":"round",
      "stroke-dasharray":circ, "stroke-dashoffset":circ, transform:"rotate(-90)"}, g);
    const chairs = [];
    for (let k = 0; k < cap; k++) {
      const a = -Math.PI/2 + k * 2 * Math.PI / cap;
      const cx = Math.cos(a) * (R + 12), cy = Math.sin(a) * (R + 12);
      el("circle", {cx, cy, r:5.5, fill:C.floor, stroke:C.line, "stroke-width":1.5}, g);
      chairs.push([x + cx, y + cy]);
    }
    const halo = el("circle", {r:ringR + 9, fill:"none", stroke:C.free, "stroke-width":3, opacity:0, filter:"url(#glow)"}, g);
    const body = el("circle", {r:R, fill:C.floor, stroke:C.free, "stroke-width":2}, g);
    const mins = el("text", {y:5, "text-anchor":"middle", class:"tmin"}, g);
    el("text", {y:ringR + 20, "text-anchor":"middle", class:"tname"}, g).textContent = `T${id} · ${cap} seats`;
    tables.set(id, {id, cap, x, y, R, g, ring, circ, body, halo, mins, chairs, party:null, total:0, left:0, dots:[]});
  }
}

// Parties
// Lay out one dot per guest in a compact party group.
function clusterOffsets(n) {
  const cols = Math.min(n, 4), rows = Math.ceil(n / 4), out = [];
  for (let i = 0; i < n; i++) {
    const r = Math.floor(i / 4), c = i % 4, inRow = r < rows - 1 ? cols : n - 4 * r;
    out.push([(c - (inRow - 1) / 2) * 12, (r - (rows - 1) / 2) * 12]);
  }
  return out;
}
// Create a party graphic with its guest dots, waiting bar and selection highlight.
function makeParty(id, size) {
  const g = el("g", {opacity:0}, layers.parties);
  const offs = clusterOffsets(size);
  const halfW = Math.min(size, 4) * 6, halfH = Math.ceil(size / 4) * 6;
  const halo = el("rect", {x:-halfW - 7, y:-halfH - 6, width:halfW * 2 + 34, height:halfH * 2 + 12, rx:10,
    fill:"none", stroke:C.hl, "stroke-width":2, opacity:0, filter:"url(#glow)"}, g);
  const dots = offs.map(([dx, dy]) => el("circle", {cx:dx, cy:dy, r:5, fill:C.guest}, g));
  el("text", {x:halfW + 8, y:5, class:"psize"}, g).textContent = size;
  const extra = el("g", {opacity:0}, g);
  el("rect", {x:60, y:-3, width:240, height:6, rx:3, fill:C.line}, extra);
  const bar = el("rect", {x:60, y:-3, width:0, height:6, rx:3, fill:C.guest}, extra);
  const wt = el("text", {x:312, y:5, class:"wtxt"}, extra);
  const p = {id, size, g, halo, dots, offs, extra, bar, wt, x:ENTRY[0], y:ENTRY[1], wait:0, barW:0, where:"buffer"};
  setPos(p, ENTRY[0], ENTRY[1]);
  parties.set(id, p);
  return p;
}
function setPos(p, x, y) { p.x = x; p.y = y; p.g.setAttribute("transform", `translate(${x},${y})`); }

// Tables
// Show the fraction of recorded dining time remaining around a table.
function setRing(t, frac) {
  t.ring.setAttribute("stroke-dashoffset", t.circ * (1 - Math.max(0, Math.min(1, frac))));
}
// Refresh a table's occupancy colour and remaining-minute label.
function paintTable(t) {
  const busy = t.party !== null;
  t.body.setAttribute("stroke", busy ? C.guest : C.free);
  t.body.setAttribute("fill", busy ? C.guest : C.floor);
  t.body.setAttribute("fill-opacity", busy ? .14 : 1);
  t.mins.textContent = busy ? `${t.left * 5}′` : "";
}

// Narration and stats
const nMain = document.getElementById("nMain"), nSub = document.getElementById("nSub");
/* Each sentence stays on screen long enough to read: the next one waits for it.
   Hold time is in real (playing) time, shortened only gently at higher speeds. */
let playClock = 0, lastSay = -1e9;
(function tickClock() {
  let last = performance.now();
  (function f(now) { if (playing) playClock += Math.min(now - last, 50); last = now; requestAnimationFrame(f); })(last);
})();
function holdTime(text) { return (1500 + text.length * 18) / Math.sqrt(speed); }
let holdNeeded = 0;
// Keep narration readable before replacing it with the next event description.
async function say(main, sub) {
  const g = gen;
  await new Promise((res, rej) => {
    (function f() {
      if (g !== gen) return rej(ABORT);
      playClock - lastSay >= holdNeeded ? res() : requestAnimationFrame(f);
    })();
  });
  nMain.style.opacity = 0; nSub.style.opacity = 0;
  await new Promise(r => setTimeout(r, 160));
  nMain.textContent = main; nSub.textContent = sub || "";
  nMain.style.opacity = 1; nSub.style.opacity = 1;
  lastSay = playClock; holdNeeded = holdTime(main + (sub || ""));
}
// Refresh the visible totals from the current replay counters.
function stats() {
  document.getElementById("sServed").textContent = served;
  document.getElementById("sLost").textContent = lost;
  document.getElementById("sRet").textContent = ret.toFixed(1);
}

// Sync to a recorded snapshot
// Match party positions and table timers to the next saved snapshot.
async function sync(snap, animate) {
  const q = [...snap.queue].sort((a, b) => b.wait - a.wait || a.id.localeCompare(b.id));
  const moves = [];
  q.forEach((s, i) => {
    let p = parties.get(s.id) || makeParty(s.id, s.size);
    p.where = "queue"; p.wait = s.wait;
    moves.push({p, x:QX, y:QY0 + i * QDY, showExtra:1, barTo: Math.min(s.wait, 12) / 12 * 240});
  });
  snap.buffer.forEach((s, j) => {
    let p = parties.get(s.id) || makeParty(s.id, s.size);
    p.where = "buffer";
    moves.push({p, x:BX0 + j * BDX, y:BY, showExtra:0, barTo:0});
  });
  // Separate completed meals from occupied tables whose timers need updating.
  const releases = [], ringUpd = [];
  for (const st of snap.tables) {
    const t = tables.get(st.id);
    if (t.party && !st.party) releases.push(t);
    else if (st.party) ringUpd.push([t, t.left, st.left]);
  }
  // Interpolate arrivals, waiting bars and departures toward the recorded snapshot.
  const run = k => {
    for (const m of moves) {
      const p = m.p;
      if (m.x0 === undefined) { m.x0 = p.x; m.y0 = p.y; m.o0 = +p.g.getAttribute("opacity"); m.e0 = +p.extra.getAttribute("opacity"); m.b0 = p.barW; }
      setPos(p, lerp(m.x0, m.x, k), lerp(m.y0, m.y, k));
      p.g.setAttribute("opacity", lerp(m.o0, 1, k));
      p.extra.setAttribute("opacity", lerp(m.e0, m.showExtra, k));
      p.barW = lerp(m.b0, m.barTo, k);
      p.bar.setAttribute("width", p.barW);
      p.bar.setAttribute("fill", mixColor(C.guest, C.lost, Math.min(1, p.wait / 10)));
      p.wt.textContent = `${p.wait * 5} min`;
    }
    for (const [t, a, b] of ringUpd) { const v = lerp(a, b, k); setRing(t, v / t.total); }
    for (const t of releases) {
      t.dots.forEach((d, i) => {
        if (!d._from) d._from = [+d.getAttribute("cx"), +d.getAttribute("cy")];
        const kk = Math.max(0, Math.min(1, k * 1.3 - i * 0.04));
        d.setAttribute("cx", lerp(d._from[0], EXIT_R[0], kk * kk));
        d.setAttribute("cy", lerp(d._from[1], EXIT_R[1], kk * kk));
        d.setAttribute("opacity", 1 - kk);
      });
      setRing(t, lerp(t.left / t.total, 0, k));
    }
  };
  if (animate) await anim(900, run); else run(1);
  for (const [t, , b] of ringUpd) { t.left = b; paintTable(t); }
  for (const t of releases) { t.dots.forEach(d => d.remove()); t.dots = []; t.party = null; t.left = 0; t.total = 0; paintTable(t); }
  return {releases, arrivals: snap.buffer.length};
}

// One seating decision
/* Lines run along the empty aisles between table rows, so they never cross another table. */
const AISLE_X = 490;
const aisleY = t => t.y < 300 ? 100 : t.y < 500 ? 270 : 470;
// Convert aisle waypoints into an SVG route with rounded corners.
function roundedPath(pts, r) {
  pts = pts.filter((q, i) => i === 0 || Math.hypot(q[0] - pts[i - 1][0], q[1] - pts[i - 1][1]) > 0.5);
  let d = `M ${pts[0][0]} ${pts[0][1]}`;
  for (let i = 1; i < pts.length - 1; i++) {
    const [x0, y0] = pts[i - 1], [x1, y1] = pts[i], [x2, y2] = pts[i + 1];
    const l1 = Math.hypot(x1 - x0, y1 - y0), l2 = Math.hypot(x2 - x1, y2 - y1);
    const rr = Math.min(r, l1 / 2, l2 / 2);
    const ax = x1 - (x1 - x0) / l1 * rr, ay = y1 - (y1 - y0) / l1 * rr;
    const bx = x1 + (x2 - x1) / l2 * rr, by = y1 + (y2 - y1) / l2 * rr;
    d += ` L ${ax} ${ay} Q ${x1} ${y1} ${bx} ${by}`;
  }
  const last = pts[pts.length - 1];
  return d + ` L ${last[0]} ${last[1]}`;
}
// Route the party-to-table line through the dining-room aisles.
function routeTo(p, c, lane) {
  const sx = p.where === "queue" ? 452 : p.x + Math.min(p.size, 4) * 6 + 30, ax = AISLE_X + lane * 5, ay = aisleY(c) + lane * 5;
  const ey = c.y - (c.R + 24) - 3;
  // party already level with the aisle: run straight along it (no tiny step)
  if (Math.abs(p.y - aisleY(c)) < 30) return roundedPath([[sx, p.y], [c.x, p.y], [c.x, ey]], 30);
  return roundedPath([[sx, p.y], [ax, p.y], [ax, ay], [c.x, ay], [c.x, ey]], 34);
}

// Animate the recorded seating choice and show its compatible alternatives.
async function seat(s) {
  const p = parties.get(s.id), t = tables.get(s.table);
  // every free table this party fits gets a soft halo
  const fitting = [...tables.values()].filter(x => !x.party && COMPAT[s.size].includes(x.cap));
  // one line per table size: same-size tables are interchangeable (the lowest-numbered free one is used)
  const cands = [];
  for (const cap of COMPAT[s.size]) {
    const free = fitting.filter(x => x.cap === cap).sort((a, b) => a.id - b.id);
    if (free.length) cands.push(free[0]);
  }
  if (!cands.includes(t)) cands.push(t);
  // Use recorded masked PPO probabilities for line widths; these lines do not choose an action.
  const lines = cands.map((c, i) => {
    const pr = s.probs ? (s.probs[actionId(s.size, c.cap)] || 0) : null;
    const w = pr === null ? 1.4 : 0.9 + pr * 1.8;
    const op = pr === null ? .55 : .25 + .55 * pr;
    const L = el("path", {d:routeTo(p, c, 0), fill:"none", stroke:C.muted,
      "stroke-width":w, "stroke-linecap":"round", "stroke-linejoin":"round", opacity:op}, layers.lines);
    const len = L.getTotalLength();
    L.setAttribute("stroke-dasharray", len); L.setAttribute("stroke-dashoffset", len);
    let T = null;
    if (pr !== null) {
      T = el("text", {x:c.x + c.R + 30, y:c.y - c.R - 24, class:"pct", opacity:0}, layers.lines);
      T.textContent = `${Math.round(pr * 100)}%`;
    }
    return {L, c, w, op, T, len};
  });
  const waited = p.where === "queue" ? `They waited ${p.wait * 5} min.` : "They had only just arrived.";
  const spare = t.cap - s.size;
  const fit = spare === 0 ? "A perfect fit: no empty seats." : `${spare} seat${spare > 1 ? "s" : ""} will stay empty.`;
  let conf = "";
  if (s.probs) {
    const pc = s.probs[s.a] || 0, pw = s.probs["11"] || 0;
    conf = ` ${policy} gave this move ${Math.round(pc * 100)}%${pw >= 0.01 ? `, and ${Math.round(pw * 100)}% to waiting instead` : ""}.`;
  }
  const nFit = fitting.length;
  // a party that has waited clearly longer, could have been seated now, but was passed over
  const freeCaps = new Set(fitting.map(x => x.cap).concat([...tables.values()].filter(x => !x.party).map(x => x.cap)));
  const myWait = p.where === "queue" ? p.wait : 0;
  const skipped = [...parties.values()]
    .filter(x => x !== p && x.where === "queue" && x.wait - myWait >= 2 && COMPAT[x.size].some(c => freeCaps.has(c)))
    .sort((a, b) => b.wait - a.wait)[0];
  let skipTxt = "";
  if (skipped) {
    skipTxt = ` The party of ${skipped.size} has waited ${skipped.wait * 5} min but is passed over`;
    if (s.probs) {
      const pk = Math.max(...COMPAT[skipped.size].map(c => s.probs[actionId(skipped.size, c)] || 0));
      skipTxt += ` (${policy} gave them ${pk < 0.01 && pk > 0 ? "under 1" : Math.round(pk * 100)}%)`;
    }
    skipTxt += ".";
    skipped.halo.setAttribute("stroke", C.lost);
  }
  await say(`Seat the party of ${s.size} at T${t.id}.`, `${waited} ${nFit} free table${nFit > 1 ? "s fit" : " fits"} them. ${fit}${conf}${skipTxt}`);
  p.g.parentNode.appendChild(p.g);
  p.halo.setAttribute("opacity", 1);
  const others = [...parties.values()].filter(x => x !== p);
  const unfit = [...tables.values()].filter(x => !fitting.includes(x));
  const dimTable = (x, v) => { x.g.setAttribute("opacity", v); x.dots.forEach(d => d.setAttribute("opacity", v)); };
  // 1. light up every table that fits, draw the options
  await anim(s.probs ? 560 : 420, k => {
    for (const f of fitting) f.halo.setAttribute("opacity", .9 * k);
    for (const x of unfit) dimTable(x, lerp(1, .28, k));
    for (const o of others) o.g.setAttribute("opacity", lerp(1, o === skipped ? .85 : .22, k));
    if (skipped) skipped.halo.setAttribute("opacity", k);
    for (const {L, T, len} of lines) { L.setAttribute("stroke-dashoffset", len * (1 - k)); if (T) T.setAttribute("opacity", k); }
  });
  // 2. commit: the chosen route and table brighten, the rest fade back
  const chosen = lines.find(o => o.c === t);
  await anim(360, k => {
    for (const o of lines) {
      if (o === chosen) { o.L.setAttribute("stroke", C.hl); o.L.setAttribute("stroke-width", lerp(o.w, 2.6, k)); o.L.setAttribute("opacity", lerp(o.op, 1, k)); o.L.setAttribute("filter", "url(#glow)"); }
      else { o.L.setAttribute("opacity", lerp(o.op, .1, k)); if (o.T) o.T.setAttribute("opacity", 1 - .7 * k); }
    }
    for (const f of fitting) {
      if (f === t) { f.halo.setAttribute("stroke", C.hl); f.halo.setAttribute("opacity", 1); f.halo.setAttribute("stroke-width", lerp(3, 5, k)); }
      else { f.halo.setAttribute("opacity", lerp(.9, .12, k)); dimTable(f, lerp(1, .45, k)); }
    }
  });
  // 3. guests walk the route to their chairs
  p.extra.setAttribute("opacity", 0);
  const starts = p.offs.map(([dx, dy]) => [p.x + dx, p.y + dy]);
  const flying = p.dots.map((d, i) => el("circle", {cx:starts[i][0], cy:starts[i][1], r:5, fill:C.guest}, layers.fly));
  p.g.remove(); parties.delete(p.id);
  const route = chosen.L, rlen = chosen.len;
  const p0 = route.getPointAtLength(0), pEnd = route.getPointAtLength(rlen);
  const n = flying.length, gap = 0.07, D = 1 - gap * (n - 1);
  // guests walk single file exactly on the line, gather at the table, then all sit down together
  const W = 0.8;
  await anim(1400 + n * 60, (k, raw) => {
    const sit = raw <= W ? 0 : ease((raw - W) / (1 - W));
    flying.forEach((d, i) => {
      const u = Math.max(0, Math.min(1, (raw / W - i * gap) / D));
      let x, y;
      if (u < 0.12) {
        const q = ease(u / 0.12);
        x = lerp(starts[i][0], p0.x, q); y = lerp(starts[i][1], p0.y, q);
      } else {
        const q = ease((u - 0.12) / 0.88), pt = route.getPointAtLength(q * rlen);
        const g = Math.max(0, (q - 0.9) / 0.1);           // small huddle at the table edge
        x = pt.x + p.offs[i][0] * 0.6 * g; y = pt.y + p.offs[i][1] * 0.6 * g;
      }
      if (sit > 0) { x = lerp(x, t.chairs[i][0], sit); y = lerp(y, t.chairs[i][1], sit); }
      d.setAttribute("cx", x); d.setAttribute("cy", y);
    });
    const fade = Math.max(0, (raw - 0.8) / 0.2);
    for (const o of lines) { o.L.setAttribute("opacity", o === chosen ? .95 * (1 - fade) : .1 * (1 - k)); if (o.T) o.T.setAttribute("opacity", (o === chosen ? 1 : .3) * (1 - fade)); }
    for (const f of fitting) f.halo.setAttribute("opacity", (f === t ? 1 : .12) * (1 - fade));
    t.halo.setAttribute("stroke-width", 4 + 1.5 * Math.sin(raw * Math.PI * 4));
    setRing(t, k);
  });
  lines.forEach(o => { o.L.remove(); if (o.T) o.T.remove(); });
  for (const f of fitting) { f.halo.setAttribute("opacity", 0); f.halo.setAttribute("stroke", C.free); f.halo.setAttribute("stroke-width", 3); }
  t.party = s.id; t.total = s.dur; t.left = s.dur; t.dots = flying;
  paintTable(t);
  served += s.size; stats();
  const dimmed = [...unfit, ...fitting.filter(f => f !== t)];
  await anim(220, k => {
    for (const x of dimmed) dimTable(x, lerp(+x.g.getAttribute("opacity") || 1, 1, k));
    for (const o of others) if (parties.has(o.id)) o.g.setAttribute("opacity", lerp(o === skipped ? .85 : .22, 1, k));
    if (skipped) skipped.halo.setAttribute("opacity", 1 - k);
  });
  dimmed.forEach(x => dimTable(x, 1));
  if (skipped) skipped.halo.setAttribute("stroke", C.hl);
  await compactQueue();
}

// Close visual gaps after a party leaves the queue or arrival area.
async function compactQueue() {
  const q = [...parties.values()].filter(x => x.where === "queue").sort((a, b) => a.y - b.y);
  const mv = q.map((x, i) => ({x, y0:x.y, y1:QY0 + i * QDY})).filter(m => Math.abs(m.y0 - m.y1) > 0.5);
  const bf = [...parties.values()].filter(x => x.where === "buffer").sort((a, b) => a.x - b.x);
  const mb = bf.map((x, j) => ({x, x0:x.x, x1:BX0 + j * BDX})).filter(m => Math.abs(m.x0 - m.x1) > 0.5);
  if (!mv.length && !mb.length) return pause(180);
  await anim(320, k => { for (const m of mv) setPos(m.x, m.x.x, lerp(m.y0, m.y1, k)); for (const m of mb) setPos(m.x, lerp(m.x0, m.x1, k), m.x.y); });
}

// Five minutes pass
// Animate recorded losses, penalties and the transition to the next five-minute snapshot.
async function endTick(tk, next) {
  const clock = document.getElementById("clock");
  const lostIds = [...tk.end.abandoned, ...tk.end.balked];
  const leaving = lostIds.map(id => parties.get(id)).filter(Boolean);
  for (const p of leaving) { p.dots.forEach(d => d.setAttribute("fill", C.lost)); parties.delete(p.id); }
  let res = {releases: [], arrivals: 0};
  let flyers = [];
  if (leaving.length) {
    const why = p => tk.end.balked.includes(p.id) ? "the queue was full" : `waiting ${p.wait * 5 + 5} min`;
    await say(leaving.length === 1 ? `A party of ${leaving[0].size} gives up and leaves.` : `${leaving.length} parties give up and leave.`,
        leaving.map(p => `Party of ${p.size}: ${why(p)}`).join(" · "));
    const tags = leaving.map(p => {
      p.halo.setAttribute("stroke", C.lost);
      const tg = el("text", {x:p.x + 70, y:p.y - 14, class:"losttag", opacity:0}, layers.fly);
      tg.textContent = tk.end.balked.includes(p.id) ? "no room, leaves" : "gives up";
      return tg;
    });
    await anim(650, (k, raw) => {
      const pulse = 0.5 + 0.5 * Math.sin(raw * Math.PI * 3);
      for (const p of leaving) {
        p.dots.forEach(d => d.setAttribute("fill", mixColor(C.guest, C.lost, Math.min(1, raw * 2))));
        p.halo.setAttribute("opacity", 0.4 + 0.6 * pulse);
        p.extra.setAttribute("opacity", 1 - k);
      }
      tags.forEach(t => t.setAttribute("opacity", Math.min(1, raw * 3)));
    });
    flyers = leaving.flatMap(p => p.offs.map(([dx, dy]) => {
      const c = el("circle", {cx:p.x + dx, cy:p.y + dy, r:5, fill:C.lost}, layers.fly);
      const slot = trayCount++;
      return {c, abandoned: tk.end.abandoned.includes(p.id), x0:p.x + dx, y0:p.y + dy, x1:TRAY_X + (slot % 46) * 8.5, y1:TRAY_Y + Math.floor(slot / 46) * 9};
    }));
    leaving.forEach(p => p.g.remove());
    var tagEls = tags;
  }
  {
    const waitEls = [...parties.values()].flatMap(p => p.dots)
      .concat(flyers.filter(f => f.abandoned).map(f => f.c));
    await rewardTokens(tk, waitEls, flyers.map(f => f.c));
  }
  // Move recorded lost guests to the tally while the next snapshot updates tables and arrivals.
  const lostAnim = anim(900, (k, raw) => {
    flyers.forEach(({c, x0, y0, x1, y1}, i) => {
      const u = ease(Math.max(0, Math.min(1, raw * 1.15 - i * 0.01)));
      c.setAttribute("cx", lerp(x0, x1, u)); c.setAttribute("cy", lerp(y0, y1, u) - Math.sin(u * Math.PI) * 30);
      c.setAttribute("r", lerp(5, 3.2, u));
    });
    if (tagEls) tagEls.forEach(t => t.setAttribute("opacity", 1 - k));
  });
  clock.textContent = clockText(tk.tick + 1);
  const syncP = next ? sync(next.start, true).then(r => { res = r; }) : Promise.resolve();
  await Promise.all([lostAnim, syncP]);
  if (tagEls) tagEls.forEach(t => t.remove());
  if (leaving.length) flashLost();
  lost += tk.end.lostPersons; ret = tk.totals.return; stats();
  const bits = [];
  if (res.releases.length) bits.push(`${listT(res.releases)} ${res.releases.length > 1 ? "free up" : "frees up"}.`);
  if (res.arrivals) bits.push(`${res.arrivals} new ${res.arrivals > 1 ? "parties arrive" : "party arrives"}.`);
  if (leaving.length) bits.push(`${leaving.length > 1 ? leaving.length + " parties give" : "A party of " + leaving[0].size + " gives"} up and ${leaving.length > 1 ? "leave" : "leaves"}.`);
  let chose = "";
  if (tk.end.legal > 1) {
    const pw = tk.end.probs ? tk.end.probs["11"] : null;
    chose = `${policy} chose to wait although a seating was possible${pw !== null && pw !== undefined ? ` (masked action probability ${Math.round(pw * 100)}%)` : ""}. `;
  }
  const waitedTxt = chose + (tk.end.waitedPersons ? `${tk.end.waitedPersons} guest${tk.end.waitedPersons > 1 ? "s" : ""} waited through those 5 minutes.` : "Nobody had to wait.");
  await say(`Five minutes pass. ${bits.join(" ") || "Nothing changes."}`, waitedTxt);
  await pause(300);
}
function listT(ts) {
  const n = ts.map(t => "T" + t.id);
  return n.length === 1 ? n[0] : n.slice(0, -1).join(", ") + " and " + n[n.length - 1];
}

function flashLost() {
  const b = document.getElementById("sLost");
  b.classList.remove("flash"); void b.offsetWidth; b.classList.add("flash");
}

// Main loop
// Restart the selected saved trajectory and replay seating and time events in order.
async function run() {
  // Invalidate pending animations before resetting the scene for the selected trace.
  const myGen = ++gen;
  const tr = DATA[`${policy}|${scen}`];
  parties = new Map(); served = 0; lost = 0; ret = 0; trayCount = 0;
  fx.innerHTML = ""; setFormula(0, 0, true);
  buildScene(); stats();
  document.getElementById("meta").textContent = `Dinner · ${SCEN[scen]} · replay of development seed ${tr.seed}` +
    (TSEED[policy] ? ` · model from training seed ${TSEED[policy]}` : "");
  document.getElementById("clock").textContent = clockText(tr.ticks[0].tick);
  lastSay = -1e9; holdNeeded = 0;
  try {
    await say(`${policy} is hosting tonight.`, RULE[policy]);
    await sync(tr.ticks[0].start, false);
    await pause(900);
    // Consume saved events only; playback never runs a model or resamples restaurant outcomes.
    for (let i = 0; i < tr.ticks.length; i++) {
      const tk = tr.ticks[i];
      for (const s of tk.seats) { await untilPlaying(); await seat(s); afterEvent(); }
      if (tk.end) { await untilPlaying(); await endTick(tk, tr.ticks[i + 1]); afterEvent(); }
    }
    await say(`Closing time. ${tr.final.served} guests seated, ${tr.final.lost} lost.`,
        `Return for this one evening: ${tr.final.return.toFixed(1)}. One evening is a single example; the report averages 800 evenings per model (2,400 per learned architecture).`);
    setPlaying(false);
    await new Promise(r => setTimeout(r, 900));
    if (gen === myGen) showSummary();
  } catch (e) { if (e !== ABORT) throw e; }
}
function afterEvent() { if (stepOnce) { stepOnce = false; setPlaying(false); } }

// End-of-evening comparison
// Mean return in the frozen final test, per demand level (200 evenings per baseline, 600 per learned group).
const FINAL_SCEN = {"off_peak":{"FIFO":-11.79,"Best-fit":-11.6,"DQN A":-12.67,"DQN C":-12.42,"PPO A":-11.65,"PPO C":-12.08,"PPO G":-11.65},"normal":{"FIFO":-30.33,"Best-fit":-29.4,"DQN A":-31.17,"DQN C":-30.73,"PPO A":-29.66,"PPO C":-30.02,"PPO G":-29.45},"peak":{"FIFO":-53.02,"Best-fit":-49.4,"DQN A":-51.84,"DQN C":-50.76,"PPO A":-50.31,"PPO C":-50.91,"PPO G":-49.86},"super_peak":{"FIFO":-87.82,"Best-fit":-80.84,"DQN A":-83.34,"DQN C":-82.9,"PPO A":-82.3,"PPO C":-85.85,"PPO G":-81.25}};
const sumEl = document.getElementById("summary");
// Animate a summary number from zero to its saved value.
function countUp(elm, to, digits, ms) {
  const t0 = performance.now();
  (function f(now) {
    const k = Math.min(1, (now - t0) / ms), e = 1 - Math.pow(1 - k, 3);
    elm.textContent = (to * e).toFixed(digits);
    if (k < 1) requestAnimationFrame(f);
  })(t0);
}
// Compare this development evening with the stored final-test means for the selected demand level.
function showSummary() {
  setPlaying(false);
  const rows = HOSTS.map(h => ({h, ...DATA[`${h}|${scen}`].final, avg: FINAL_SCEN[scen][h]}));
  const bestRet = Math.max(...rows.map(r => r.return)), bestAvg = Math.max(...rows.map(r => r.avg));
  const maxAbs = Math.max(...rows.map(r => Math.abs(r.return)));
  document.getElementById("sumTitle").textContent = `Closing time · Dinner · ${SCEN[scen]}`;
  document.getElementById("sumAvgHead").textContent = `Final test avg, ${SCEN[scen]}`;
  const body = document.getElementById("sumRows");
  body.innerHTML = "";
  rows.forEach((r, i) => {
    const row = document.createElement("div");
    row.className = "srow" + (r.h === policy ? " me" : "");
    row.style.animationDelay = `${120 + i * 70}ms`;
    const gap = r.return - bestRet;
    row.innerHTML = `<span class="sh">${r.h}</span><b class="n sv">0</b><b class="n sl">0</b>
      <b class="n sr${r.return === bestRet ? " best" : ""}">0.0</b>
      <span class="n sg">${gap === 0 ? "–" : gap.toFixed(1)}</span>
      <b class="n sa${r.avg === bestAvg ? " best" : ""}">${r.avg.toFixed(2)}</b>`;
    body.appendChild(row);
    setTimeout(() => {
      countUp(row.querySelector(".sv"), r.served, 0, 900);
      countUp(row.querySelector(".sl"), r.lost, 0, 900);
      countUp(row.querySelector(".sr"), r.return, 1, 900);
    }, 200 + i * 70);
  });
  sumEl.hidden = false;
  requestAnimationFrame(() => sumEl.classList.add("open"));
}
function hideSummary() { sumEl.classList.remove("open"); setTimeout(() => { sumEl.hidden = true; }, 250); }
document.getElementById("sumClose").onclick = hideSummary;
document.getElementById("sumReplay").onclick = () => { hideSummary(); setPlaying(true); run(); };
document.getElementById("bCompare").onclick = showSummary;
sumEl.addEventListener("click", e => { if (e.target === sumEl) hideSummary(); });

// Controls
const bPlay = document.getElementById("bPlay");
const PAUSE_D = "M7 5h3.5v14H7zM13.5 5H17v14h-3.5z", PLAY_D = "M8 5l11 7-11 7z";
// Keep playback state, the button icon and its accessible label in agreement.
function setPlaying(v) {
  playing = v;
  document.getElementById("playIcon").setAttribute("d", v ? PAUSE_D : PLAY_D);
  bPlay.setAttribute("aria-label", v ? "Pause" : "Play");
}
bPlay.onclick = () => setPlaying(!playing);
document.getElementById("bStep").onclick = () => { stepOnce = true; setPlaying(true); };
document.getElementById("bReset").onclick = () => { setPlaying(true); run(); };
// Build a choice group and keep its pressed state aligned with the selected value.
function seg(groupId, items, get, set) {
  const g = document.getElementById(groupId), btns = [];
  for (const it of items) {
    if (it === null) { const d = document.createElement("span"); d.className = "sep"; g.appendChild(d); continue; }
    const [val, label] = it;
    const b = document.createElement("button");
    b.textContent = label; b.type = "button";
    b.onclick = () => { set(val); btns.forEach(([v, bb]) => bb.setAttribute("aria-pressed", v === get())); };
    g.appendChild(b); btns.push([val, b]);
  }
  btns.forEach(([v, bb]) => bb.setAttribute("aria-pressed", v === get()));
}
seg("gSpeed", [[0.5,"0.5×"],[1,"1×"],[2,"2×"],[4,"4×"]], () => speed, v => { speed = v; });
const _setHost = v => { policy = v; hideSummary(); setPlaying(true); run(); };
seg("gPolicy", [["Best-fit","Best-fit"],["FIFO","FIFO"],null,["DQN A","DQN A"],["DQN C","DQN C"],null,["PPO A","PPO A"],["PPO C","PPO C"],["PPO G","PPO G"]], () => policy, _setHost);
seg("gScen", [["off_peak","Off-Peak"],["normal","Normal"],["peak","Peak"],["super_peak","Super Peak"]], () => scen, v => { scen = v; hideSummary(); setPlaying(true); run(); });
document.addEventListener("keydown", e => {
  if (e.target.tagName === "BUTTON" && (e.key === " " || e.key === "Enter")) return;
  if (e.key === " ") { e.preventDefault(); setPlaying(!playing); }
  else if (e.key === "ArrowRight") { stepOnce = true; setPlaying(true); }
  else if (e.key === "r" || e.key === "R") { hideSummary(); setPlaying(true); run(); }
  else if (e.key === "c" || e.key === "C") { sumEl.hidden ? showSummary() : hideSummary(); }
  else if (e.key === "Escape") hideSummary();
});
run();
})();
