/* 정적 비교 뷰어. export_static.py가 구운 data/*.json만 읽는다 (서버 없음). */
"use strict";

const $ = (id) => document.getElementById(id);
const state = { manifest: null, data: null, qi: 0 };

const fmt = (v, d = 4) => (v === null || v === undefined ? "—" : Number(v).toFixed(d));
const labelOf = (key) =>
  (state.manifest.models.find((m) => m.key === key) || {}).label || key;
const isBaseline = (key) =>
  ((state.manifest.models.find((m) => m.key === key) || {}).baseline) === true;

async function boot() {
  state.manifest = await fetch("data/manifest.json").then((r) => r.json());

  const subsetSel = $("subset");
  state.manifest.subsets.forEach((s) => {
    subsetSel.add(new Option(`${s.config} (쿼리 ${s.n_queries}개)`, s.config));
  });

  subsetSel.onchange = () => loadSubset(subsetSel.value);
  $("criterion").onchange = render;
  $("query").onchange = () => { state.qi = +$("query").value; render(); };
  $("prev").onclick = () => step(-1);
  $("next").onclick = () => step(1);
  $("rand").onclick = () => {
    state.qi = Math.floor(Math.random() * state.data.queries.length);
    render();
  };

  const g = state.manifest.generated_with;
  $("footer-note").textContent =
    `서브셋당 쿼리 ${g.queries}개를 '${g.select}' 기준으로 골라 top-${g.topk}까지 미리 계산했습니다. ` +
    `정확도는 서브셋 전체(쿼리 전부) 평균이고, 속도·메모리는 같은 조건에서 측정했습니다.`;

  renderSummary();
  await loadSubset(state.manifest.subsets[0].config);
}

function step(delta) {
  const n = state.data.queries.length;
  state.qi = (state.qi + delta + n) % n;
  render();
}

async function loadSubset(config) {
  state.data = await fetch(`data/${config}.json`).then((r) => r.json());
  state.qi = 0;

  const sel = $("query");
  sel.innerHTML = "";
  state.data.queries.forEach((q, i) => {
    const attrs = q.attrs.slice(0, 3).join(", ") || "속성 없음";
    sel.add(new Option(`#${q.idx}  ${q.category}  [${attrs}]`, i));
  });
  renderSummary();
  render();
}

/* ---------------------------------------------------------- 요약표 */
function renderSummary() {
  const config = state.data ? state.data.config : state.manifest.subsets[0].config;
  const crit = $("criterion") ? $("criterion").value : "coarse";
  const rows = state.manifest.summary;
  const cols = [
    [`${crit} R@1`, (r) => r.accuracy[config]?.[`${crit}_recall@1`], true],
    [`${crit} R@10`, (r) => r.accuracy[config]?.[`${crit}_recall@10`], true],
    [`${crit} mAP@10`, (r) => r.accuracy[config]?.[`${crit}_map@10`], true],
    ["1000장당(초)", (r) => r.sec_per_1000, false],
    ["속도 (DreamSim=1)", (r) => r.speed_ratio, false],
    ["최대 VRAM(MB)", (r) => r.peak_vram_mb, false],
    ["모델 크기(MB)", (r) => r.model_size_mb, false],
    ["차원", (r) => r.embed_dim, false],
  ];

  // 정확도 열은 최고값을 굵게 표시한다
  const best = {};
  cols.forEach(([name, get, isAcc]) => {
    if (!isAcc) return;
    best[name] = Math.max(...rows.map((r) => get(r) ?? -1));
  });

  const head = `<thead><tr><th>모델</th>${cols
    .map(([n]) => `<th>${n}</th>`)
    .join("")}</tr></thead>`;

  const body = rows
    .map((r) => {
      const cells = cols
        .map(([name, get, isAcc]) => {
          const v = get(r);
          if (name.startsWith("속도")) {
            if (r.is_baseline) return `<td>1.00 (기준)</td>`;
            if (v == null) return `<td>—</td>`;
            return `<td>${v.toFixed(2)}× ${v > 1 ? "느림" : "빠름"}</td>`;
          }
          if (v == null) return `<td>—</td>`;
          if (isAcc) {
            const cls = v === best[name] ? ' class="best"' : "";
            return `<td${cls}>${fmt(v)}</td>`;
          }
          return `<td>${Number.isInteger(v) ? v : fmt(v, 1)}</td>`;
        })
        .join("");
      const tag = r.is_baseline ? '<span class="tag">기준</span>' : "";
      return `<tr class="${r.is_baseline ? "baseline" : ""}">
                <td>${r.label}${tag}</td>${cells}</tr>`;
    })
    .join("");

  $("summary").innerHTML = head + `<tbody>${body}</tbody>`;

  const measured = rows.find((r) => r.measured);
  $("summary-note").textContent = measured
    ? `정확도는 ${config} 전체 평균입니다. 속도·메모리는 ${measured.config}/query ` +
      `${measured.n_images}장, batch ${measured.batch_size}, ${measured.gpu} 기준입니다. ` +
      `절대 시간은 머신 상태에 따라 달라지므로 배수를 보세요.`
    : "속도·메모리 측정값이 없습니다.";
}

/* ---------------------------------------------------------- 쿼리 + 결과 */
function render() {
  const q = state.data.queries[state.qi];
  const crit = $("criterion").value;
  $("query").value = state.qi;

  $("q-img").src = q.img;
  $("q-meta").innerHTML = `
    <dt>서브셋</dt><dd>${state.data.config}</dd>
    <dt>쿼리</dt><dd>#${q.idx}</dd>
    <dt>카테고리</dt><dd>${q.category}</dd>
    <dt>속성</dt><dd>${q.attrs.join(", ") || "—"}</dd>
    <dt>갤러리 정답 수</dt>
    <dd>${q.n_relevant[crit].toLocaleString()}개 / ${state.data.gallery_size.toLocaleString()}장</dd>`;

  const models = state.data.models
    .slice()
    .sort((a, b) => (isBaseline(b) ? 1 : 0) - (isBaseline(a) ? 1 : 0));

  let baseMetrics = null;
  let baseRows = null;
  const blocks = models.map((key) => {
    const res = q.results[key];
    const m = res.metrics[crit];
    const base = isBaseline(key) && baseMetrics === null;
    if (base) {
      baseMetrics = m;
      baseRows = new Set(res.rows);
    }

    const chips = [];
    if (base) chips.push('<span class="chip base">기준</span>');
    chips.push(`<span class="chip">R@1 ${m["recall@1"].toFixed(0)}</span>`);
    chips.push(`<span class="chip">R@10 ${m["recall@10"].toFixed(0)}</span>`);
    chips.push(`<span class="chip">mAP@10 ${fmt(m["map@10"], 3)}</span>`);
    chips.push(`<span class="chip">top-10 정답 ${m["hits@10"]}개</span>`);
    if (!base && baseMetrics) {
      const diff = m["map@10"] - baseMetrics["map@10"];
      const cls = Math.abs(diff) < 1e-9 ? "" : diff > 0 ? " up" : " down";
      const text = Math.abs(diff) < 1e-9 ? "기준과 같음" : (diff > 0 ? "+" : "") + diff.toFixed(3);
      chips.push(`<span class="chip${cls}">${text}</span>`);
      const overlap = res.rows.filter((r) => baseRows.has(r)).length;
      chips.push(`<span class="chip">기준과 겹침 ${overlap}/${res.rows.length}</span>`);
    }

    const cards = res.rows
      .map((row, i) => {
        const g = state.data.gallery[row];
        const hit = res.hits[crit][i];
        return `<div class="card ${hit ? "hit" : ""}">
            <img loading="lazy" src="${g.img}" alt="${g.label}">
            <div class="meta"><b>${hit ? "●" : "○"} ${i + 1}위</b>
              <span>${res.scores[i].toFixed(3)}</span></div>
            <div class="label">${g.label}</div>
          </div>`;
      })
      .join("");

    return `<div class="model">
        <div class="model-head"><span class="model-name">${labelOf(key)}</span>
          ${chips.join("")}</div>
        <div class="row">${cards}</div>
      </div>`;
  });

  $("results").innerHTML = blocks.join("");
}

boot().catch((err) => {
  document.body.insertAdjacentHTML(
    "afterbegin",
    `<p style="padding:20px;color:#dc2626">데이터를 불러오지 못했습니다: ${err}</p>`
  );
});
