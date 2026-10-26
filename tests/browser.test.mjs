import assert from "node:assert/strict";
import test from "node:test";
import { displayName, exportCsv, filterRows, historyThrough, readState } from "../web/data.mjs";
import { chartValue, forecastChart } from "../web/charts.mjs";

const state = { date:"2025-06-02", query:"", role:"all", view:"watchlist", sort:"performance_rank", direction:1 };
const rows = [
  { pitcher:1, name:"Colomé, Alex", forecast_date:"2025-06-02", start_share:0, performance_rank:2, performance_watchlist:true, alert:false, y_bf:0, p_k_B:0.25 },
  { pitcher:2, name:"Cole, Gerrit", forecast_date:"2025-06-02", start_share:1, performance_rank:1, performance_watchlist:true, alert:true, y_bf:100, p_k_B:0.3 },
  { pitcher:1, name:"Colomé, Alex", forecast_date:"2025-06-09", start_share:0, performance_rank:20, performance_watchlist:false, alert:false, y_bf:40, p_k_B:0.2 },
];

test("watchlist selection retains no-followup forecasts and fixed ranks", () => {
  assert.deepEqual(filterRows(rows,state).map(row => row.pitcher), [2,1]);
  const changed = rows.map(row => ({ ...row,y_bf:999 }));
  assert.deepEqual(filterRows(changed,state).map(row => row.pitcher), [2,1]);
});

test("accent-insensitive search and role filter work together", () => {
  assert.equal(displayName("Colomé, Alex"), "Alex Colomé");
  assert.deepEqual(filterRows(rows,{ ...state,query:"alex colome",role:"reliever" }).map(row => row.pitcher), [1]);
  assert.equal(filterRows(rows,{ ...state,query:"alex colome",role:"starter" }).length,0);
});

test("history never includes forecasts after the selected cutoff", () => {
  assert.equal(historyThrough(rows,rows[0]).length,1);
});

test("exports do not leak hidden outcomes", () => {
  const item = { ...rows[1], y_bf:100,observed_k:0.4,resolution:"confirmed" };
  assert.ok(!exportCsv([item],false).includes("observed_k"));
  assert.ok(!exportCsv([item],false).includes("confirmed"));
  assert.ok(exportCsv([item],true).includes("observed_k"));
});

test("CSV quotes names and neutralizes spreadsheet formulas in strings", () => {
  assert.ok(exportCsv([{ ...rows[0],name:'A, "B"' }],false).includes('"A, ""B"""'));
  assert.ok(exportCsv([{ ...rows[0],name:"=1+1" }],false).includes("'=1+1"));
});

test("deep links retain the chosen forecast and reject unsupported views", () => {
  const s = readState("?season=2024&date=all&at=2024-05-06&pitcher=1&view=alerts&outcomes=1");
  assert.equal(s.at,"2024-05-06");
  assert.equal(s.pitcher,1);
  assert.equal(s.view,"alerts");
  assert.equal(s.reveal,true);
  assert.equal(readState("?view=bogus&season=2026").season,"2025");
  assert.equal(readState("?view=misses").reveal,true);
  assert.equal(readState("?view=alerts").date,"all");
});

test("pitch histories retain physical units and never use outcome series", () => {
  const html = forecastChart([{ forecast_date:"2025-06-02",fb_velo_cur:95,observed_k_minus_bb:0.99 }],true,"velocity");
  assert.ok(html.includes("95.0 mph"));
  assert.ok(!html.includes("observed-series"));
  assert.equal(chartValue(0.5,"usage"),"50.0%");
});

test("missing pitch measurements do not become a zero-valued history", () => {
  const html = forecastChart([{ forecast_date:"2025-06-02",fb_velo_cur:null }],false,"velocity");
  assert.ok(html.includes("No measurements available"));
  assert.ok(!html.includes("<svg"));
});
