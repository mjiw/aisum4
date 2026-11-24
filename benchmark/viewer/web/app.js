/* 임베딩 모델 검색 비교 — 정적 뷰어.
 * export_static.py가 구운 data/bundle.js(window.VIEWER_DATA)만 읽는다. 서버가 없다.
 * 화면 상태는 주소의 # 뒤에 담겨 링크로 그대로 공유된다.
 */
"use strict";

const D = window.VIEWER_DATA;
const $ = (id) => document.getElementById(id);
const CRITS = [["coarse", "coarse"], ["fine", "fine"]];
const PICK_KEY = "aisum.compare.picked";

const MODEL = Object.fromEntries((D ? D.models : []).map((m) => [m.key, m]));
const BASE = D ? D.meta.baseline : null;
const SUBSETS = D ? Object.keys(D.subsets) : [];

const S = {
  view: "compare",       // 처음 열면 검색 결과 화면부터 보여준다
  scope: "all",
  crit: "coarse",
  subset: SUBSETS[0],
  qi: 0,
  picked: null,          // Set — 기준 모델 아래에 보여줄 모델
  // 표는 처음부터 순위대로: 정확도는 R@1 높은 순, 속도는 1000장 처리 빠른 순
  sort: { acc: { col: "r1", dir: "desc" }, perf: { col: "t1000", dir: "asc" } },
};
const VIEWS = ["compare", "summary", "settings"];

/* ---------------------------------------------------------------- 포맷 */
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const isNum = (v) => typeof v === "number" && Number.isFinite(v);
const pct = (v) => (isNum(v) ? (v * 100).toFixed(1) : "—");
const fixed = (v, d = 1) => (isNum(v) ? v.toFixed(d) : "—");
const mb = (v) => (!isNum(v) ? "—" : v >= 1024 ? `${(v / 1024).toFixed(2)} GB` : `${v.toFixed(0)} MB`);
const params = (v) => (!isNum(v) ? "—" : v >= 1e9 ? `${(v / 1e9).toFixed(2)}B` : `${(v / 1e6).toFixed(0)}M`);
const label = (key) => (MODEL[key] ? MODEL[key].label : key);

function deltaPP(v, b) {
  if (!isNum(v) || !isNum(b)) return "";
  const d = (v - b) * 100;
  if (Math.abs(d) < 0.05) return `<em class="d">±0.0</em>`;
  return `<em class="d${d > 0 ? " up" : ""}">${d > 0 ? "▲" : "▼"}${Math.abs(d).toFixed(1)}</em>`;
}

/* ---------------------------------------------------------------- 주소(#) 상태 */
function readHash() {
  const [view, query] = location.hash.replace(/^#\/?/, "").split("?");
  const p = new URLSearchParams(query || "");
  if (VIEWS.includes(view)) S.view = view;
  if (p.get("crit") === "coarse" || p.get("crit") === "fine") S.crit = p.get("crit");
  if (p.has("scope") && (p.get("scope") === "all" || SUBSETS.includes(p.get("scope"))))
    S.scope = p.get("scope");
  if (p.has("subset") && SUBSETS.includes(p.get("subset"))) S.subset = p.get("subset");
  if (p.has("q")) S.qi = Math.max(0, parseInt(p.get("q"), 10) || 0);
  if (p.has("m")) S.picked = new Set(p.get("m").split(",").filter((k) => MODEL[k] && k !== BASE));
}

function writeHash() {
  const p = new URLSearchParams();
  if (S.view !== "settings") p.set("crit", S.crit);
  if (S.view === "summary") p.set("scope", S.scope);
  else if (S.view === "compare") {
    p.set("subset", S.subset);
    p.set("q", String(S.qi));
    p.set("m", [...S.picked].join(","));
  }
  history.replaceState(null, "", `#${S.view}?${p.toString()}`);
}

function loadPicked() {
  if (S.picked) return;
  try {
    const saved = JSON.parse(localStorage.getItem(PICK_KEY) || "null");
    if (Array.isArray(saved)) S.picked = new Set(saved.filter((k) => MODEL[k] && k !== BASE));
  } catch (_) { /* 저장소를 못 쓰는 환경이면 기본값으로 */ }
  if (!S.picked) S.picked = new Set(D.models.filter((m) => !m.baseline).map((m) => m.key));
}

function savePicked() {
  try { localStorage.setItem(PICK_KEY, JSON.stringify([...S.picked])); } catch (_) { /* 무시 */ }
}

/* ---------------------------------------------------------------- 공통 */
function seg(el, items, current, onPick) {
  el.innerHTML = items.map(([value, text]) =>
    `<button type="button" data-v="${esc(value)}" class="${value === current ? "is-on" : ""}">${esc(text)}</button>`
  ).join("");
  el.onclick = (e) => {
    const b = e.target.closest("button[data-v]");
    if (b) onPick(b.dataset.v);
  };
}

function setView(view) {
  S.view = view;
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("is-on", t.dataset.view === view));
  VIEWS.forEach((v) => { $(`view-${v}`).hidden = v !== view; });
  render();
}

function render() {
  if (S.view === "summary") renderSummary();
  else if (S.view === "settings") renderSettings();
  else renderCompare();
  writeHash();
}

function modelCell(m, withBranch = true) {
  const links = Object.entries(m.links || {}).map(([k, url]) => {
    const name = { paper: "논문", hf: "HF", code: "코드", blog: "블로그", bench: "LookBench" }[k] || k;
    return `<a href="${esc(url)}" target="_blank" rel="noopener">${name}</a>`;
  });
  const tag = m.baseline ? '<span class="tag">기준</span>' : "";
  const br = withBranch
    ? `<span class="br" title="평가에 쓴 브랜치">${esc(m.branch)}${m.commit ? "@" + esc(m.commit) : ""}</span>` : "";
  return `<div class="mcell"><span class="mname">${esc(m.label)}${tag}</span>
    <span class="mlinks">${br}${links.join("")}</span></div>`;
}

/* 표 정렬 = 순위. 기준 모델(DreamSim)도 제자리에서 순위를 받는다(줄 배경으로 구분).
 * 값이 없는 모델은 맨 아래. 같은 값이면 같은 순위(1, 2, 2, 4). 같은 열을 다시 누르면 방향이 바뀐다. */
function sortRows(rows, table, cols) {
  const st = S.sort[table];
  const col = st && cols.find((c) => c.id === st.col);
  const val = (r) => (col ? col.get(r) : null);
  const out = rows.slice();
  if (col) {
    out.sort((a, b) => {
      const va = val(a), vb = val(b);
      if (!isNum(va) && !isNum(vb)) return 0;
      if (!isNum(va)) return 1;
      if (!isNum(vb)) return -1;
      return st.dir === "asc" ? va - vb : vb - va;
    });
  }
  let prev = null, rank = 0;
  out.forEach((r, i) => {
    const v = val(r);
    if (!isNum(v)) { r.rank = null; return; }
    if (v !== prev) { rank = i + 1; prev = v; }
    r.rank = rank;
  });
  return out;
}

function rankCell(r) {
  return `<td class="rk${r.rank === 1 ? " top" : ""}">${r.rank ?? "—"}</td>`;
}

function headRow(table, cols) {
  const st = S.sort[table];
  return `<thead><tr><th class="rk" title="누른 열 기준 순위">순위</th><th class="l">모델</th>${cols.map((c) => {
    const arr = st && st.col === c.id ? (st.dir === "asc" ? "↑" : "↓") : "";
    return `<th data-col="${c.id}" title="${esc(c.tip || "누르면 정렬")}">${esc(c.name)} <span class="arr">${arr}</span></th>`;
  }).join("")}</tr></thead>`;
}

function bindSort(el, table, cols) {
  el.querySelector("thead").onclick = (e) => {
    const th = e.target.closest("th[data-col]");
    if (!th) return;
    const col = cols.find((c) => c.id === th.dataset.col);
    const st = S.sort[table];
    const first = col.lowerBetter ? "asc" : "desc";
    S.sort[table] = st && st.col === col.id
      ? { col: col.id, dir: st.dir === "asc" ? "desc" : "asc" }
      : { col: col.id, dir: first };
    renderSummary();
  };
}

/* ---------------------------------------------------------------- 성능 요약 */
function scopeName(scope) {
  return scope === "all" ? "전체" : scope;
}

function renderSummary() {
  const scopes = ["all", ...SUBSETS];
  seg($("sum-scope"), scopes.map((s) => [s, scopeName(s)]), S.scope, (v) => { S.scope = v; render(); });
  seg($("sum-crit"), CRITS, S.crit, (v) => { S.crit = v; render(); });

  const sum = D.summary[S.scope] || {};
  const rows = D.models.map((m) => ({ m, s: sum[m.key] ? sum[m.key][S.crit] : null,
                                      vs: sum[m.key] && sum[m.key].vs ? sum[m.key].vs[S.crit] : null,
                                      n: sum[m.key] ? sum[m.key].n_queries : null }));
  const base = rows.find((r) => r.m.baseline);
  const nq = rows.find((r) => r.n)?.n;

  renderTiles(rows, base);
  renderAccuracy(rows, base);
  renderPerf();

  $("acc-hint").textContent = nq
    ? `${scopeName(S.scope)} 쿼리 ${nq.toLocaleString()}개 전부로 계산 · 단위 % · ▲▼는 DreamSim 대비 %p · 열 이름을 누르면 그 지표 기준으로 순위가 바뀝니다`
    : "이 서브셋 결과가 없습니다";
}

function renderTiles(rows, base) {
  const have = rows.filter((r) => r.s);
  const tiles = [];
  const best = (get) => have.reduce((a, r) => (!a || get(r) > get(a) ? r : a), null);
  const b1 = best((r) => r.s["recall@1"]);
  if (b1) tiles.push(["R@1 1위", b1.m.label,
    `${pct(b1.s["recall@1"])}%` + (base && base.s && !b1.m.baseline
      ? ` · DreamSim ${pct(base.s["recall@1"])}%` : "")]);
  const bm = best((r) => r.s["map@10"]);
  if (bm) tiles.push(["mAP@10 1위", bm.m.label,
    `${pct(bm.s["map@10"])}` + (base && base.s && !bm.m.baseline ? ` · DreamSim ${pct(base.s["map@10"])}` : "")]);
  if (base && base.s) {
    const others = have.filter((r) => !r.m.baseline);
    const better = others.filter((r) => r.s["recall@1"] > base.s["recall@1"]).length;
    tiles.push(["DreamSim보다 R@1 높은 모델", `${better} / ${others.length}`,
      `DreamSim R@1 ${pct(base.s["recall@1"])}%`]);
  }
  const lat = D.models.filter((m) => isNum(m.perf.latency_ms_b1));
  if (lat.length) {
    const fast = lat.reduce((a, m) => (m.perf.latency_ms_b1 < a.perf.latency_ms_b1 ? m : a));
    const bp = MODEL[BASE] && MODEL[BASE].perf.latency_ms_b1;
    tiles.push(["쿼리 1장 가장 빠름", fast.label,
      `${fixed(fast.perf.latency_ms_b1)} ms` + (isNum(bp) && fast.key !== BASE ? ` · DreamSim ${fixed(bp)} ms` : "")]);
  }
  $("tiles").innerHTML = tiles.map(([k, v, s]) =>
    `<div class="tile"><div class="tile-k">${esc(k)}</div><div class="tile-v">${esc(v)}</div><div class="tile-s">${esc(s)}</div></div>`
  ).join("");
}

function renderAccuracy(rows, base) {
  const cols = [
    { id: "r1", name: "R@1", get: (r) => r.s && r.s["recall@1"] },
    { id: "r10", name: "R@10", get: (r) => r.s && r.s["recall@10"] },
    { id: "map", name: "mAP@10", get: (r) => r.s && r.s["map@10"] },
    { id: "vs", name: "DreamSim 대비 (승 · 무 · 패)", tip: "쿼리별 AP@10 비교. 누르면 승률로 정렬",
      get: (r) => (r.vs ? r.vs.win / (r.vs.win + r.vs.tie + r.vs.loss) : null) },
  ];
  const max = Object.fromEntries(cols.slice(0, 3).map((c) =>
    [c.id, Math.max(...rows.map((r) => c.get(r)).filter(isNum))]));

  const body = sortRows(rows, "acc", cols).map((r) => {
    if (!r.s) {
      return `<tr class="is-missing">${rankCell(r)}<td class="l">${modelCell(r.m)}</td><td colspan="${cols.length}">결과 없음 (아직 실행 전)</td></tr>`;
    }
    const cells = cols.slice(0, 3).map((c) => {
      const v = c.get(r);
      const w = isNum(v) && max[c.id] > 0 ? (v / max[c.id]) * 100 : 0;
      const d = r.m.baseline ? "" : deltaPP(v, base && c.get(base));
      return `<td><span class="num${v === max[c.id] ? " best" : ""}"><span><b>${pct(v)}</b>${d}</span>
        <span class="bar"><i style="width:${w.toFixed(1)}%"></i></span></span></td>`;
    });
    let vs = "<td></td>";
    if (r.vs) {
      const n = r.vs.win + r.vs.tie + r.vs.loss;
      const w = (x) => ((x / n) * 100).toFixed(1);
      vs = `<td><span class="wtl"><span class="wtl-bar"><span class="w" style="width:${w(r.vs.win)}%"></span>
        <span class="t" style="width:${w(r.vs.tie)}%"></span><span class="l" style="width:${w(r.vs.loss)}%"></span></span>
        <span class="wtl-txt"><b>${r.vs.win}</b> · ${r.vs.tie} · ${r.vs.loss}</span></span></td>`;
    } else if (r.m.baseline) {
      vs = `<td><span class="wtl-txt">기준</span></td>`;
    }
    return `<tr class="${r.m.baseline ? "is-base" : ""}">${rankCell(r)}<td class="l">${modelCell(r.m)}</td>${cells.join("")}${vs}</tr>`;
  }).join("");

  const el = $("acc-table");
  el.innerHTML = headRow("acc", cols) + `<tbody>${body}</tbody>`;
  bindSort(el, "acc", cols);
}

function tokensOf(p) {
  if (isNum(p.llm_tokens_per_image)) return p.llm_tokens_per_image;
  return isNum(p.vision_tokens_per_image) ? p.vision_tokens_per_image : null;
}

function renderPerf() {
  const bp = (MODEL[BASE] || {}).perf || {};
  const cols = [
    { id: "t1000", name: "1000장 처리 (초)", lowerBetter: true, get: (r) => r.p.sec_per_1000,
      fmt: (v, r) => `<b>${fixed(v, 2)}</b>` + (isNum(v) && isNum(bp.sec_per_1000) && !r.m.baseline
        ? `<em class="d${v < bp.sec_per_1000 ? " up" : ""}">×${(v / bp.sec_per_1000).toFixed(2)}</em>` : "") },
    { id: "lat", name: "쿼리 1장 (ms)", lowerBetter: true, get: (r) => r.p.latency_ms_b1,
      fmt: (v, r) => `<b>${fixed(v, 1)}</b>` + (isNum(v) && isNum(bp.latency_ms_b1) && !r.m.baseline
        ? `<em class="d${v < bp.latency_ms_b1 ? " up" : ""}">×${(v / bp.latency_ms_b1).toFixed(2)}</em>` : "") },
    { id: "tok", name: "이미지 토큰", lowerBetter: true, get: (r) => tokensOf(r.p),
      fmt: (v, r) => isNum(v) ? `<b>${Math.round(v).toLocaleString()}</b>` +
        (isNum(r.p.llm_tokens_per_image) ? `<em class="d" title="VLM: 언어 모델 입력 길이(평균). 비전 패치 ${Math.round(r.p.vision_tokens_per_image || 0).toLocaleString()}개">LLM</em>` : "") : "—" },
    { id: "vram", name: "최대 VRAM", lowerBetter: true, get: (r) => r.p.peak_vram_mb, fmt: (v) => mb(v) },
    { id: "params", name: "파라미터", lowerBetter: true, get: (r) => r.p.model_params, fmt: (v) => params(v) },
    { id: "size", name: "모델 크기", lowerBetter: true, get: (r) => r.p.model_size_mb, fmt: (v) => mb(v) },
    { id: "dim", name: "차원", lowerBetter: true, get: (r) => r.p.embed_dim, fmt: (v) => (isNum(v) ? String(v) : "—") },
    { id: "index", name: "인덱스 (6만 장)", lowerBetter: true, get: (r) => r.p.gallery_60k_mb, fmt: (v) => mb(v) },
    { id: "load", name: "로딩 (초)", lowerBetter: true, get: (r) => r.p.load_sec, fmt: (v) => fixed(v, 1) },
  ];
  const rows = D.models.map((m) => ({ m, p: m.perf || {} }));
  const body = sortRows(rows, "perf", cols).map((r) => {
    if (!isNum(r.p.sec_per_1000)) {
      return `<tr class="is-missing">${rankCell(r)}<td class="l">${modelCell(r.m)}</td><td colspan="${cols.length}">측정 전</td></tr>`;
    }
    const cells = cols.map((c) => `<td>${c.fmt(c.get(r), r)}</td>`).join("");
    return `<tr class="${r.m.baseline ? "is-base" : ""}">${rankCell(r)}<td class="l">${modelCell(r.m)}</td>${cells}</tr>`;
  }).join("");
  const el = $("perf-table");
  el.innerHTML = headRow("perf", cols) + `<tbody>${body}</tbody>`;
  bindSort(el, "perf", cols);

  const measured = D.models.map((m) => m.perf).filter((p) => isNum(p.sec_per_1000));
  const gpus = [...new Set(measured.map((p) => p.gpu).filter(Boolean))];
  const batches = [...new Set(measured.map((p) => p.batch_size))].sort((a, b) => a - b);
  const n = measured[0] ? measured[0].n_images : null;
  $("perf-hint").textContent = measured.length
    ? `${gpus.join(", ") || "GPU"} · ${measured[0].config} 쿼리 ${n}장 · 배치 ${batches.join("/")} (모델 설정값) · ×는 DreamSim 대비, 굵으면 더 빠름` +
      (gpus.length > 1 ? " · ⚠ 측정 GPU가 서로 다릅니다" : "")
    : "측정값이 없습니다";
}

/* ---------------------------------------------------------------- 실험 설정 */
const uniq = (xs) => [...new Set(xs.filter((x) => x !== null && x !== undefined && x !== ""))];
const n0 = (v) => (isNum(v) ? v.toLocaleString() : "—");

function renderCommon() {
  const ev = D.meta.eval || {};
  const ds = D.meta.dataset;
  const pf = ev.perf || {};
  const measured = D.models.map((m) => m.perf || {}).filter((p) => isNum(p.sec_per_1000));
  const caps = uniq(D.models.flatMap((m) => Object.values((m.settings || {}).runs || {}).map((r) => r.vram)));
  const env = [
    uniq(measured.map((p) => p.gpu)).join(", ") || "GPU",
    `torch ${uniq(measured.map((p) => p.torch)).join(", ") || "—"}`,
    `transformers ${uniq(measured.map((p) => p.transformers)).join(", ") || "—"}`,
    caps.length ? `VRAM 상한 ${caps.map((c) => `${Math.round(c * 100)}%`).join(" / ")} (프로세스당)` : "",
  ].filter(Boolean).join(" · ");
  const subRows = Object.entries(ev.subsets || {}).map(([k, s]) =>
    `<tr><td class="l">${esc(k)}</td><td>${n0(s.n_queries)}</td><td>${n0(s.n_subset_gallery)}</td><td>${n0(s.n_gallery)}</td></tr>`
  ).join("");

  const items = [
    ["데이터셋", `<a href="${esc(ds.url)}" target="_blank" rel="noopener">${esc(ds.name)}</a>
      · revision <code>${esc((ds.revision || "").slice(0, 7))}</code>`],
    ["서브셋", `<table class="mini"><thead><tr><th class="l">서브셋</th><th>쿼리</th><th>서브셋 갤러리</th>
      <th title="서브셋 갤러리 + noise">전체 갤러리</th></tr></thead><tbody>${subRows}</tbody></table>`],
    ["갤러리", `서브셋 갤러리 + 공용 방해 이미지(noise) ${n0(ev.noise)}장 · noise는 항상 오답으로 칩니다`],
    ["정답 기준", `<b>coarse</b> 같은 카테고리면 정답 · <b>fine</b> 카테고리와 속성 집합이 완전히 같아야 정답 (${esc(ev.fine_mode || "")})`],
    ["검색", `임베딩을 L2 정규화한 뒤 코사인 유사도로 갤러리 전체와 비교해 상위 ${ev.topk ?? 10}개 (근사 검색 없음)`],
    ["지표", `R@1 · R@10 · mAP@10 — 서브셋의 모든 쿼리로 계산하고, '전체'는 네 서브셋 쿼리를 합쳐 계산`],
    ["실행 환경", `${esc(env)} · 한 번에 한 모델만 GPU를 씁니다`],
    ["속도 측정", `${esc(pf.config || "")} 쿼리 ${n0(pf.n_images)}장을 모델별 배치로 ${pf.repeats ?? "—"}번 처리한 중앙값
      (워밍업 ${pf.warmup_batches ?? "—"}배치) · 쿼리 1장 = 배치 1로 ${n0(pf.latency_n)}장 잰 중앙값
      · 이미지 토큰 = ${n0(pf.token_n)}장 평균 · 모든 모델이 같은 이미지로 잽니다`],
  ];
  $("common").innerHTML = `<dl class="kv">${items.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}</dl>`;
}

function batchCell(st) {
  const runs = st.runs || {};
  const groups = {};
  Object.entries(runs).forEach(([sub, r]) => { (groups[r.batch] = groups[r.batch] || []).push(sub); });
  const keys = Object.keys(groups);
  let evalHtml = "—";
  if (keys.length === 1) evalHtml = `<b>${esc(keys[0])}</b>`;
  else if (keys.length > 1) {
    evalHtml = keys.map((b) => `<div><b>${esc(b)}</b> <span class="muted">${groups[b].map(esc).join(", ")}</span></div>`).join("");
  }
  const perf = isNum(st.perf_batch) ? `<div class="muted">속도 측정 ${st.perf_batch}</div>` : "";
  return evalHtml + perf;
}

function ckptCell(st) {
  const id = st.checkpoint || "";
  const ex = st.extra || {};
  let main;
  if (/^[\w.-]+\/[\w.-]+$/.test(id)) {
    main = `<a href="https://huggingface.co/${esc(id)}" target="_blank" rel="noopener">${esc(id)}</a>`;
  } else if (id) {
    main = esc(id);
  } else {
    main = esc([ex.source, ex.checkpoint || ex.filename].filter(Boolean).join(" · ")) || '<span class="muted">패키지 기본 가중치</span>';
  }
  return main + (st.revision ? `<div class="muted mono">@${esc(st.revision)}</div>` : "");
}

function fmtVal(v) {
  if (v === null) return "없음";
  if (typeof v === "object") return esc(JSON.stringify(v).replace(/[{}"]/g, "").replace(/,/g, ", ").replace(/:/g, ": "));
  return esc(v);
}

function renderSettings() {
  renderCommon();
  const head = `<thead><tr><th>모델</th><th>입력 해상도</th><th>배치</th><th>임베딩 차원</th>
    <th>풀링</th><th>가중치</th><th>코드</th><th>비고</th></tr></thead>`;
  const body = D.models.map((m) => {
    const st = m.settings || {};
    if (!Object.keys(st.runs || {}).length) {
      return `<tr class="is-missing"><td>${modelCell(m, false)}</td><td colspan="7">결과 없음 (아직 실행 전)</td></tr>`;
    }
    const dim = isNum(st.embed_dim) ? `<b>${st.embed_dim}</b>` +
      (isNum(st.full_dim) && st.full_dim !== st.embed_dim ? `<div class="muted">MRL로 줄임 (원래 ${st.full_dim})</div>` : "") : "—";
    const code = `<span class="mono">${esc(m.branch)}${m.commit ? "@" + esc(m.commit) : ""}</span>` +
      (st.adapter ? `<div class="muted">어댑터 '${esc(st.adapter)}' — 기준 브랜치 위에서 실행</div>` : "");
    const extra = Object.entries(st.extra || {});
    const notes = [
      m.note ? `<div>${esc(m.note)}</div>` : "",
      st.run_args && st.run_args.length ? `<div><code>${esc(st.run_args.join(" "))}</code></div>` : "",
      extra.length ? `<details><summary>모델 상세 ${extra.length}개</summary><dl class="kv small">${
        extra.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${fmtVal(v)}</dd>`).join("")}</dl></details>` : "",
    ].join("") || '<span class="muted">—</span>';
    return `<tr class="${m.baseline ? "is-base" : ""}">
      <td>${modelCell(m, false)}</td>
      <td>${esc(st.input || "—")}</td>
      <td>${batchCell(st)}</td>
      <td>${dim}</td>
      <td>${esc(st.pooling || "—")}</td>
      <td class="ck">${ckptCell(st)}</td>
      <td>${code}</td>
      <td class="nt">${notes}</td>
    </tr>`;
  }).join("");
  $("set-table").innerHTML = head + `<tbody>${body}</tbody>`;
}

/* ---------------------------------------------------------------- 검색 결과 */
function subsetData() { return D.subsets[S.subset]; }

function renderCompare() {
  const sub = subsetData();
  if (!sub) { $("rows").innerHTML = `<div class="empty">결과가 없습니다</div>`; return; }
  if (S.qi >= sub.queries.length) S.qi = 0;

  const sel = $("cmp-subset");
  if (sel.options.length !== SUBSETS.length) {
    sel.innerHTML = SUBSETS.map((k) => {
      const s = D.subsets[k];
      return `<option value="${esc(k)}">${esc(k)} · ${s.queries.length}/${s.n_queries.toLocaleString()} 쿼리</option>`;
    }).join("");
    sel.onchange = () => { S.subset = sel.value; S.qi = 0; render(); };
  }
  sel.value = S.subset;
  seg($("cmp-crit"), CRITS, S.crit, (v) => { S.crit = v; render(); });
  $("qpos").textContent = `${S.qi + 1} / ${sub.queries.length}`;

  renderChips(sub);
  renderStrip(sub);
  renderQuery(sub);
  renderRows(sub);
}

function renderChips(sub) {
  const avail = new Set(sub.models);
  const chips = D.models.map((m) => {
    if (m.baseline) return `<span class="chip is-fixed" title="항상 맨 위에 표시">${esc(m.label)} · 기준</span>`;
    if (!avail.has(m.key)) return `<button class="chip" disabled title="이 서브셋 결과가 없습니다">${esc(m.label)}</button>`;
    return `<button class="chip${S.picked.has(m.key) ? " is-on" : ""}" data-k="${esc(m.key)}">${esc(m.label)}</button>`;
  });
  const el = $("model-chips");
  el.innerHTML = chips.join("");
  el.onclick = (e) => {
    const b = e.target.closest("button[data-k]");
    if (!b) return;
    const k = b.dataset.k;
    if (S.picked.has(k)) S.picked.delete(k); else S.picked.add(k);
    savePicked();
    render();
  };
}

function renderStrip(sub) {
  const el = $("qstrip");
  if (el.dataset.subset !== S.subset) {
    el.innerHTML = sub.queries.map((q, i) =>
      `<button class="qthumb" data-i="${i}" title="#${q.idx} ${esc(q.cat)}"><img loading="lazy" decoding="async" src="${esc(q.img)}" alt=""></button>`
    ).join("");
    el.dataset.subset = S.subset;
    el.onclick = (e) => {
      const b = e.target.closest("button[data-i]");
      if (b) { S.qi = +b.dataset.i; render(); }
    };
  }
  el.querySelectorAll(".qthumb").forEach((b, i) => b.classList.toggle("is-on", i === S.qi));
  const cur = el.children[S.qi];
  if (cur) cur.scrollIntoView({ block: "nearest", inline: "center" });
}

function renderQuery(sub) {
  const q = sub.queries[S.qi];
  const attrs = q.attrs.length
    ? `<span class="attrs">${q.attrs.map((a) => `<span class="attr">${esc(a)}</span>`).join("")}</span>`
    : "—";
  $("qcard").innerHTML = `
    <div class="qimg"><img src="${esc(q.img)}" alt="쿼리 이미지"></div>
    <dl class="qmeta">
      <dt>쿼리</dt><dd>${esc(S.subset)} #${q.idx}</dd>
      <dt>카테고리</dt><dd>${esc(q.cat)}</dd>
      <dt>속성</dt><dd>${attrs}</dd>
      <dt>정답 수</dt><dd>coarse ${q.n_rel.coarse.toLocaleString()} · fine ${q.n_rel.fine.toLocaleString()}</dd>
      <dt>갤러리</dt><dd>${sub.n_gallery.toLocaleString()}장</dd>
    </dl>`;
}

function renderRows(sub) {
  const q = sub.queries[S.qi];
  const avail = new Set(sub.models);
  const order = D.models.map((m) => m.key)
    .filter((k) => avail.has(k) && (k === BASE || S.picked.has(k)));
  const base = q.res[BASE];
  const baseRank = new Map(base ? base.rows.map((row, i) => [row, i + 1]) : []);
  const shownOthers = order.filter((k) => k !== BASE);

  const blocks = order.map((key) => {
    const r = q.res[key];
    const hits = r.hits[S.crit];
    const nHit = [...hits].filter((h) => h === "1").length;
    const isBase = key === BASE;
    const ap = r.ap[S.crit];

    const stats = [];
    stats.push(hits[0] === "1" ? `<span class="stat ok">1위 <b>정답</b></span>` : `<span class="stat">1위 <b>오답</b></span>`);
    stats.push(`<span class="stat">top-10 정답 <b>${nHit}</b>개</span>`);
    let apText = `<span class="stat">AP@10 <b>${ap.toFixed(3)}</b>`;
    if (!isBase && base) {
      const d = ap - base.ap[S.crit];
      apText += Math.abs(d) < 5e-4 ? `<em class="d">±0</em>` : `<em class="d${d > 0 ? " up" : ""}">${d > 0 ? "▲" : "▼"}${Math.abs(d).toFixed(3)}</em>`;
    }
    stats.push(apText + "</span>");
    if (!isBase && base) {
      const overlap = r.rows.filter((row) => baseRank.has(row)).length;
      stats.push(`<span class="stat">DreamSim과 겹침 <b>${overlap}</b>/${r.rows.length}</span>`);
    }

    const cards = r.rows.map((row, i) => {
      const g = sub.gallery[row] || { img: "", cat: "?", attrs: [] };
      const hit = hits[i] === "1";
      let mark = "";
      if (!isBase && baseRank.has(row)) {
        mark = `<span class="ds" title="DreamSim ${baseRank.get(row)}위">D${baseRank.get(row)}</span>`;
      } else if (isBase && shownOthers.length) {
        const n = shownOthers.filter((k) => q.res[k].rows.includes(row)).length;
        mark = n ? `<span class="common" title="아래 모델 중 ${n}개가 같은 이미지를 찾음">공통 ${n}</span>` : "";
      }
      const tip = `${g.cat}${g.attrs.length ? " · " + g.attrs.join(", ") : ""} · 유사도 ${r.scores[i].toFixed(3)}`;
      return `<figure class="res${hit ? " hit" : ""}" title="${esc(tip)}">
          <div class="thumb"><img loading="lazy" decoding="async" src="${esc(g.img)}" alt="${esc(g.cat)}">
            <span class="rank">${i + 1}</span>${hit ? '<span class="okmark">✓</span>' : ""}</div>
          <figcaption><span class="cat">${esc(g.cat)}</span>${mark}</figcaption>
          <figcaption><span>${r.scores[i].toFixed(3)}</span></figcaption>
        </figure>`;
    }).join("");

    return `<div class="mrow${isBase ? " is-base" : ""}">
        <div class="mrow-head"><span class="mname">${esc(label(key))}</span>${isBase ? '<span class="tag">기준</span>' : ""}
          ${stats.join("")}</div>
        <div class="grid10">${cards}</div>
      </div>`;
  });

  if (!shownOthers.length) {
    blocks.push(`<div class="empty">위에서 비교할 모델을 고르면 DreamSim 아래에 나란히 표시됩니다.</div>`);
  }
  $("rows").innerHTML = blocks.join("");
}

function step(delta) {
  const n = subsetData().queries.length;
  S.qi = (S.qi + delta + n) % n;
  render();
}

/* ---------------------------------------------------------------- 시작 */
function boot() {
  if (!D) throw new Error("data/bundle.js를 불러오지 못했습니다");
  readHash();
  loadPicked();

  const ds = D.meta.dataset;
  $("sub").textContent = `${ds.name} · DreamSim 기준 · ${D.meta.generated} 생성`;
  $("foot").innerHTML =
    `데이터셋 <a href="${esc(ds.url)}" target="_blank" rel="noopener">${esc(ds.name)}</a>` +
    ` (revision ${esc((ds.revision || "").slice(0, 7))}) · 갤러리 = 서브셋 갤러리 + 공용 noise 풀` +
    ` · 정확도는 서브셋 전체 쿼리, 검색 결과 화면은 서브셋당 ${D.meta.queries_per_subset}개 쿼리(절반은 모델 간 결과가 갈리는 쿼리, 절반은 무작위)` +
    ` · 모든 모델은 같은 정답 판정 코드로 계산했습니다.`;

  document.querySelectorAll(".tab").forEach((t) => { t.onclick = () => setView(t.dataset.view); });
  $("prev").onclick = () => step(-1);
  $("next").onclick = () => step(1);
  $("rand").onclick = () => { S.qi = Math.floor(Math.random() * subsetData().queries.length); render(); };
  $("pick-all").onclick = () => {
    S.picked = new Set(D.models.filter((m) => !m.baseline).map((m) => m.key)); savePicked(); render();
  };
  $("pick-none").onclick = () => { S.picked = new Set(); savePicked(); render(); };

  document.addEventListener("keydown", (e) => {
    if (S.view !== "compare" || e.altKey || e.ctrlKey || e.metaKey) return;
    if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
    if (e.key === "ArrowLeft") { step(-1); e.preventDefault(); }
    if (e.key === "ArrowRight") { step(1); e.preventDefault(); }
  });
  window.addEventListener("hashchange", () => { readHash(); setView(S.view); });

  setView(S.view);
}

try {
  boot();
} catch (err) {
  document.body.insertAdjacentHTML("afterbegin",
    `<p style="padding:20px">데이터를 불러오지 못했습니다: ${esc(err.message || err)}</p>`);
  console.error(err);
}
