import { PAGE_SIZE, displayName, exportCsv, filterRows, historyThrough, readState } from "./data.mjs";
import { HISTORY_METRICS, chartValue, forecastChart } from "./charts.mjs";

const $ = selector => document.querySelector(selector);
const escape = value => String(value ?? "").replace(/[&<>"']/g, ch => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;" })[ch]);
const pct = value => value == null ? "—" : `${(value * 100).toFixed(1)}%`;
const signed = (value, digits = 1) => value == null ? "—" : `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(digits)}`;
const number = value => Number(value).toLocaleString("en-US");
const dateLabel = (value, year = false) => new Date(`${value}T12:00:00Z`).toLocaleDateString("en-US", { month:"short", day:"numeric", ...(year ? { year:"numeric" } : {}), timeZone:"UTC" });
const roleLabel = row => row.start_share >= 0.5 ? "Starter" : "Reliever";
const resolutionLabel = value => ({ confirmed:"Confirmed", false_alert:"False alert", unresolved:"Unresolved" })[value] || "No alert";
const titles = { watchlist:"Performance watchlist", all:"All eligible pitchers", alerts:"Experimental alerts", misses:"Largest forecast misses" };
let state = readState(location.search);
let manifest, rows = [], filtered = [], current = null, controller;
let historyMetric = "forecast";
const bundledElement = document.getElementById("saved-study");
const bundled = bundledElement ? JSON.parse(bundledElement.textContent) : null;

function horizonEnd(date) {
  const end = new Date(`${date}T00:00:00Z`);
  end.setUTCDate(end.getUTCDate() + 29);
  return end.toISOString().slice(0,10);
}

async function fetchJson(path, signal) {
  if (bundled && Object.hasOwn(bundled,path)) return bundled[path];
  const response = await fetch(path, { signal });
  if (!response.ok) throw new Error(`Saved data returned HTTP ${response.status}. Rebuild the browser export and try again.`);
  return response.json();
}

function persist(push = false) {
  const params = new URLSearchParams({ season:state.season, date:state.date || "all", view:state.view });
  if (state.pitcher) params.set("pitcher", state.pitcher);
  if (state.at) params.set("at", state.at);
  if (state.reveal) params.set("outcomes","1");
  if (state.query) params.set("q",state.query);
  if (state.role !== "all") params.set("role",state.role);
  const url = `${location.pathname}?${params}`;
  if (url !== `${location.pathname}${location.search}`) history[push ? "pushState" : "replaceState"](null,"",url);
}

function renderControls() {
  const dates = manifest.seasons[state.season].dates;
  $("#season").value = state.season;
  $("#date").innerHTML = `<option value="">All forecast dates</option>${dates.map(date => `<option value="${date}">${dateLabel(date)}</option>`).join("")}`;
  $("#date").value = state.date;
  $("#outcomes").checked = state.reveal;
  $("#search").value = state.query;
  $("#role").value = state.role;
  const index = dates.indexOf(state.date);
  $("#previous-date").disabled = index <= 0;
  $("#next-date").disabled = index < 0 || index >= dates.length - 1;
  document.querySelectorAll("[data-view]").forEach(button => button.setAttribute("aria-pressed",String(button.dataset.view === state.view)));
}

async function loadSeason() {
  controller?.abort();
  controller = new AbortController();
  const signal = controller.signal;
  $("#loading").hidden = false;
  $("#explorer").hidden = true;
  $("#load-error").hidden = true;
  try {
    if (!manifest) manifest = await fetchJson("data/manifest.json",signal);
    const loaded = await fetchJson(`data/${state.season}.json`,signal);
    if (signal.aborted) return;
    rows = loaded.data.map(values => Object.fromEntries(loaded.columns.map((key,index) => [key,values[index]])));
    const dates = manifest.seasons[state.season].dates;
    if (state.date === "all") state.date = "";
    else if (!dates.includes(state.date)) state.date = dates.at(-1);
    $("#loading").hidden = true;
    $("#explorer").hidden = false;
    render();
  } catch (error) {
    if (error.name === "AbortError") return;
    $("#loading").hidden = true;
    $("#load-error").hidden = false;
    $("#error-message").textContent = error.message;
  }
}

function tableColumns() {
  const columns = [["performance_rank","Rank"],["name","Pitcher"],["p_k_B","K%"],["p_bb_B","BB%"],["k_minus_bb_B","K−BB%"]];
  if (!state.date) columns.push(["forecast_date","Forecast"]);
  if (state.view === "alerts") columns.push([null,"Signal"]);
  if (state.reveal) columns.push(["observed_k_minus_bb","Later K−BB%"]);
  if (state.view === "misses") columns.push(["error_pp","Miss (pp)"]);
  return columns;
}

function renderTable() {
  const columns = tableColumns();
  $("#table-head").innerHTML = `<tr>${columns.map(([key,title]) => `<th scope="col"${key ? ` aria-sort="${state.sort === key ? state.direction === 1 ? "ascending" : "descending" : "none"}"` : ""}>${key ? `<button data-sort="${key}">${title}</button>` : title}</th>`).join("")}</tr>`;
  const page = filtered.slice(state.page * PAGE_SIZE,(state.page + 1) * PAGE_SIZE);
  $("#table-body").innerHTML = page.map(row => {
    const selected = current && row.pitcher === current.pitcher && row.forecast_date === current.forecast_date;
    const cells = columns.map(([key]) => {
      if (key === "performance_rank") return `<td>${row.performance_rank}</td>`;
      if (key === "name") return `<td><button class="pitcher-button" data-pitcher="${row.pitcher}" data-at="${row.forecast_date}"${selected ? ' aria-current="true"' : ""}>${escape(displayName(row.name))}<small>${escape(row.team)} · ${roleLabel(row)}${row.repertoire_fallback ? " · Fallback" : ""}</small></button></td>`;
      if (key === "forecast_date") return `<td>${dateLabel(row.forecast_date)}</td>`;
      if (key === "error_pp") return `<td>${row.error_pp.toFixed(1)}</td>`;
      if (key === "observed_k_minus_bb") return `<td class="later-cell">${pct(row.observed_k_minus_bb)}<small>${row.y_bf ? `${number(row.y_bf)} BF` : "No follow-up"}</small></td>`;
      if (key == null) return `<td class="alert-cell">${row.alert_direction === "improvement" ? "Improvement" : "Decline"} ${signed(row.change_delta * 100)}pp${state.reveal ? `<br><span class="badge ${row.resolution}">${resolutionLabel(row.resolution)}</span>` : ""}</td>`;
      return `<td${key === "k_minus_bb_B" ? ' class="primary-number"' : ""}>${pct(row[key])}</td>`;
    }).join("");
    return `<tr${selected ? ' class="selected"' : ""}>${cells}</tr>`;
  }).join("");
  $("#empty-results").hidden = filtered.length > 0;
  const pages = Math.max(1,Math.ceil(filtered.length / PAGE_SIZE));
  $("#result-count").textContent = filtered.length ? `${number(state.page * PAGE_SIZE + 1)}–${number(Math.min((state.page + 1) * PAGE_SIZE,filtered.length))} of ${number(filtered.length)} forecasts` : "0 matching forecasts";
  $("#page-number").textContent = `${state.page + 1} / ${pages}`;
  $("#previous-page").disabled = state.page === 0;
  $("#next-page").disabled = state.page + 1 >= pages;
  $("#export").disabled = filtered.length === 0;
}

function measurement(label, value, delta, unit) {
  return `<div class="measurement-row"><span>${label}</span><span>${value == null ? "—" : value.toFixed(1)} ${unit}</span><span>${signed(delta)} ${unit}</span></div>`;
}

function renderDetail() {
  const root = $("#detail");
  if (!current) { root.innerHTML = `<div class="empty"><h2>Select a forecast.</h2><p>A pitcher’s projection, measurements, and history will appear here.</p></div>`; return; }
  const row = current, past = historyThrough(rows,row);
  const metric = HISTORY_METRICS[historyMetric];
  const chartOutcomes = state.reveal && historyMetric === "forecast";
  const historyControls = `<label class="sr-only" for="history-metric">History measurement</label><select id="history-metric">${Object.entries(HISTORY_METRICS).map(([key,definition]) => `<option value="${key}"${key === historyMetric ? " selected" : ""}>${definition.label}</option>`).join("")}</select>`;
  const baseline = row.chg_baseline_is_ps ? "prior season" : "days 31–120";
  const stats = `<div class="projection-values"><div class="main-value"><strong>${pct(row.k_minus_bb_B)}</strong><small>K% minus BB%</small></div><div><strong>${pct(row.p_k_B)}</strong><small>Strikeouts / BF</small></div><div><strong>${pct(row.p_bb_B)}</strong><small>Walks / BF</small></div></div>`;
  let outcome = "";
  if (state.reveal) outcome = `<div class="outcome-block"><span class="small-label">Observed · ${dateLabel(row.forecast_date)}–${dateLabel(horizonEnd(row.forecast_date))}</span><p>${row.y_bf ? `<strong>${pct(row.observed_k_minus_bb)} K−BB%</strong> · ${pct(row.observed_k)} K · ${pct(row.observed_bb)} BB<br>${number(row.y_bf)} batters faced${row.y_bf < 50 ? " · Small follow-up sample" : ""}` : "No subsequent appearances in this 30-day window."}</p></div>`;
  const flag = row.repertoire_fallback ? `<p class="flag-note"><strong>Repertoire fallback.</strong> Current usage is outside a training range (${escape(row.repertoire_reasons.replaceAll("use_","").replaceAll("_cur",""))}). B and C use A’s projection.</p>` : "";
  const status = { already_evident:"The direction was already evident in recent results.", not_yet_evident:"The direction was not yet evident in the recent-results comparison.", insufficient_past_results:"Too little past exposure for the recent-results comparison." }[row.recent_results_status];
  const alert = row.alert ? `<p class="alert-note"><strong>Experimental ${row.alert_direction} alert: ${signed(row.change_delta * 100)}pp versus B.</strong> ${status}${state.reveal ? ` <span class="badge ${row.resolution}">${resolutionLabel(row.resolution)}</span>${row.resolution === "unresolved" ? " Fewer than 50 future BF." : " Under the study’s outcome definition."}` : ""}</p>` : row.alert_candidate ? `<p class="alert-note">Change signal met the threshold but was suppressed by the 30-day cooldown.</p>` : "";
  const comparisons = [ ["A","Results"],["B","Current pitches"],["C","Recent changes"] ].map(([model,label]) => `<tr${model === "B" ? ' class="chosen-model"' : ""}><td>${model} · ${label}</td><td>${pct(row[`p_k_${model}`])}</td><td>${pct(row[`p_bb_${model}`])}</td><td>${pct(row[`k_minus_bb_${model}`])}</td></tr>`).join("");
  const mix = [["fb","Fastball"],["cut","Cutter"],["brk","Breaking"],["off","Offspeed"],["other","Other"]].map(([key,label]) => {
    const usage = row[`use_${key}_cur`], delta = row[`chg_use_${key}`];
    return `<div class="mix-row"><span>${label}</span><div class="mix-bar"><span style="width:${Math.max(0,Math.min(100,(usage || 0) * 100))}%"></span></div><span>${usage == null ? "—" : `${(usage * 100).toFixed(0)}%`}</span><span>${signed(delta == null ? null : delta * 100)}pp</span></div>`;
  }).join("");
  root.innerHTML = `<div class="detail-heading"><div><p class="eyebrow">Pitcher / ${state.season}</p><h2>${escape(displayName(row.name))}</h2></div><button class="text-button" id="copy-link">Copy link ↗</button></div><p class="detail-meta">${escape(row.team)} · ${roleLabel(row)} · Forecast ${dateLabel(row.forecast_date,true)}</p><div class="projection"><div class="projection-label"><span>B · Current pitches</span><span>Next 30 days</span></div>${stats}<p class="minor-note">Point forecasts. Individual uncertainty intervals are not estimated.</p>${outcome}${flag}${alert}</div>
    <section class="detail-section"><h3>What each model sees</h3><p class="section-copy">A common baseline; two additions to the evidence.</p><table class="model-table"><thead><tr><th scope="col">Model</th><th scope="col">K%</th><th scope="col">BB%</th><th scope="col">K−BB%</th></tr></thead><tbody>${comparisons}</tbody></table></section>
    <section class="detail-section"><h3>Forecast and pitch history</h3><p class="section-copy">Through ${dateLabel(row.forecast_date)}. Pitch measurements use trailing 30-day windows.</p>${historyControls}<div class="chart-legend"><span><i></i>${historyMetric === "forecast" ? "B forecast" : `${metric.label} (${metric.unit})`}</span>${chartOutcomes ? '<span><i class="observed"></i>Observed next 30 days</span>' : ""}</div>${forecastChart(past,state.reveal,historyMetric)}<p class="minor-note">${chartOutcomes ? "Observed following-month windows overlap." : "Measurements after the selected date are excluded."}</p><details class="history-data"><summary>View history as a table</summary><table><thead><tr><th scope="col">Forecast date</th><th scope="col">${historyMetric === "forecast" ? "B K−BB%" : metric.label}</th>${chartOutcomes ? '<th scope="col">Observed</th><th scope="col">Future BF</th>' : ""}</tr></thead><tbody>${past.map(point => `<tr><td>${dateLabel(point.forecast_date)}</td><td>${chartValue(point[metric.field],historyMetric)}</td>${chartOutcomes ? `<td>${pct(point.observed_k_minus_bb)}</td><td>${number(point.y_bf)}</td>` : ""}</tr>`).join("")}</tbody></table></details></section>
    <section class="detail-section"><h3>Pitch measurements</h3><p class="section-copy">Last 30 days · ${number(row.n_pitches_cur)} tracked pitches, ${number(row.fb_n_cur ?? 0)} fastballs. Changes versus ${baseline} (${row.chg_base_pitches == null ? "unknown" : number(row.chg_base_pitches)} comparison pitches).</p><div class="measurement-row header"><span>Fastball</span><span>Current</span><span>Change</span></div>${measurement("Velocity",row.fb_velo_cur,row.chg_fb_velo,"mph")}${measurement("Vertical movement",row.fb_ivb_cur,row.chg_fb_ivb,"in")}${measurement("Arm-side movement",row.fb_hb_cur,row.chg_fb_hb,"in")}<p class="minor-note">Measurements summarize the fastball group. Missing values remain unreported.</p></section>
    <section class="detail-section"><h3>Pitch mix</h3><p class="section-copy">Current share and change from ${baseline}.</p>${mix}</section><div class="detail-actions"><button id="export-history" class="text-button">Export this pitcher’s history ↓</button></div>`;
}

function renderStudy() {
  const season = manifest.seasons[state.season], report = season.report;
  const central = target => {
    const result = report.targets[target].C_minus_B;
    return `<span>${signed(result.per_bf,6)} [${signed(result.ci95[0],6)}, ${signed(result.ci95[1],6)}]</span>`;
  };
  const summary = season.watchlist.alerts[0];
  const rates = [ ["A","Results"],["B","Current pitches"],["C","Recent changes"] ].map(([model,label]) => `<tr${model === "B" ? ' class="chosen-model"' : ""}><td>${model} · ${label}</td><td>${report.targets.k.models[model].loss_per_bf.toFixed(6)}</td><td>${report.targets.bb.models[model].loss_per_bf.toFixed(6)}</td></tr>`).join("");
  $("#study-results").innerHTML = `<div class="study-heading"><h3>${state.season} · ${season.label}</h3><p>Log loss per batter faced · lower is better</p></div><table class="model-table"><thead><tr><th scope="col">Model</th><th scope="col">Strikeouts</th><th scope="col">Walks</th></tr></thead><tbody>${rates}</tbody></table><p class="comparison-note"><strong>Recent changes versus current pitches (C − B).</strong><br>Strikeouts: ${central("k")}<br>Walks: ${central("bb")}<br><small>Estimate [95% pitcher-bootstrap interval]. Positive means C has higher loss.</small></p><p class="comparison-note"><strong>Experimental alerts:</strong> ${summary.alerts} total · ${summary.confirmed} confirmed · ${summary.false_alerts} false · ${summary.unresolved} unresolved. Unresolved alerts remain in the total.</p><p class="coverage-note">${number(season.rows)} eligible forecasts · ${number(report.test_rows)} with follow-up · ${number(season.watchlist.eligible_no_followup)} without follow-up. Training: ${state.season === "2025" ? "2021–2024, with settings frozen on 2024" : "2021–2023; settings selected on this 2024 validation"}. Study fingerprint: ${manifest.spec_sha256.slice(0,12)}.</p>`;
}

function render(push = false) {
  filtered = filterRows(rows,state);
  state.page = Math.max(0,Math.min(state.page,Math.ceil(filtered.length / PAGE_SIZE) - 1));
  current = filtered.find(row => row.pitcher === state.pitcher && row.forecast_date === state.at) || filtered.find(row => row.pitcher === state.pitcher) || filtered[0] || null;
  state.pitcher = current?.pitcher || null;
  state.at = current?.forecast_date || null;
  renderControls();
  $("#results-title").textContent = titles[state.view];
  $("#period-label").textContent = `${state.season} / ${state.date ? dateLabel(state.date) : "All forecast dates"}`;
  const descriptions = {
    watchlist:"Ten highest projected K% minus BB% each date, using B. Filters retain the original study ranks.",
    all:"Every eligible forecast, including pitchers with no subsequent appearances.",
    alerts:"C differs from B by at least two percentage points after sample and cooldown rules. These signals are experimental.",
    misses:"Largest absolute gap between B’s projected and observed K% minus BB%, with at least 50 future BF. Outcomes are revealed.",
  };
  $("#view-description").textContent = descriptions[state.view];
  renderTable(); renderDetail(); renderStudy(); persist(push);
}

function download(data,name) {
  const url = URL.createObjectURL(new Blob([exportCsv(data,state.reveal)],{ type:"text/csv;charset=utf-8" }));
  const anchor = document.createElement("a");
  anchor.href = url; anchor.download = name; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url),1000);
}

$("#season").addEventListener("change",event => { state.season = event.target.value; state.date = ["alerts","misses"].includes(state.view) ? "all" : null; state.pitcher = null; state.at = null; state.page = 0; persist(true); loadSeason(); });
$("#date").addEventListener("change",event => { state.date = event.target.value; state.at = null; state.page = 0; render(true); });
for (const [id,step] of [["previous-date",-1],["next-date",1]]) $( `#${id}`).addEventListener("click",() => { const dates = manifest.seasons[state.season].dates; state.date = dates[dates.indexOf(state.date) + step]; state.at = null; state.page = 0; render(true); });
$("#outcomes").addEventListener("change",event => { state.reveal = event.target.checked; if (!state.reveal && state.view === "misses") { state.view = "all"; state.sort = "performance_rank"; state.direction = 1; } if (!state.reveal && state.sort === "observed_k_minus_bb") state.sort = "performance_rank"; state.page = 0; render(true); });
$("#search").addEventListener("input",event => { state.query = event.target.value; state.page = 0; render(); });
$("#role").addEventListener("change",event => { state.role = event.target.value; state.page = 0; render(true); });
$("#clear-filters").addEventListener("click",() => { state.query = ""; state.role = "all"; state.page = 0; render(); $("#search").focus(); });
$("#retry").addEventListener("click",loadSeason);
$("#detail").addEventListener("change",event => { if (event.target.id === "history-metric") { historyMetric = event.target.value; renderDetail(); } });
$(".brand").addEventListener("click",event => { event.preventDefault(); state = readState(""); persist(true); loadSeason(); });
$("#export").addEventListener("click",() => download(filtered,`pitcher-${state.season}-${state.view}-${state.date || "season"}.csv`));
for (const [id,step] of [["previous-page",-1],["next-page",1]]) $(`#${id}`).addEventListener("click",() => { state.page += step; render(); });
document.addEventListener("click",async event => {
  const view = event.target.closest("[data-view]");
  if (view) { state.view = view.dataset.view; state.page = 0; state.at = null; state.sort = state.view === "misses" ? "error_pp" : state.view === "alerts" ? "forecast_date" : "performance_rank"; state.direction = state.view === "misses" ? -1 : 1; if (["alerts","misses"].includes(state.view)) state.date = ""; else if (!state.date) state.date = manifest.seasons[state.season].dates.at(-1); if (state.view === "misses") state.reveal = true; render(true); }
  const sort = event.target.closest("[data-sort]");
  if (sort) { const key = sort.dataset.sort; state.direction = state.sort === key ? -state.direction : ["name","performance_rank","forecast_date"].includes(key) ? 1 : -1; state.sort = key; state.page = 0; render(); }
  const pitcher = event.target.closest("[data-pitcher]");
  if (pitcher) { state.pitcher = Number(pitcher.dataset.pitcher); state.at = pitcher.dataset.at; render(true); if (matchMedia("(max-width:900px)").matches) $("#detail").scrollIntoView({ block:"start" }); }
  if (event.target.closest("#export-history") && current) download(historyThrough(rows,current),`pitcher-${current.pitcher}-through-${current.forecast_date}.csv`);
  if (event.target.closest("#copy-link")) {
    try { await navigator.clipboard.writeText(location.href); $("#copy-link").textContent = "Link copied"; $("#announcement").textContent = "Forecast link copied."; }
    catch { $("#announcement").textContent = "Copy the forecast link from your address bar."; $("#copy-link").textContent = "Use address bar"; }
  }
});
window.addEventListener("popstate",() => { const previous = state.season; state = readState(location.search); if (state.date === "all") state.date = ""; if (previous !== state.season) loadSeason(); else render(); });
loadSeason();
