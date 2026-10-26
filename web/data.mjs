export const PAGE_SIZE = 20;
export const VIEWS = ["watchlist", "all", "alerts", "misses"];

export function displayName(name) {
  const parts = name.split(",");
  return parts.length > 1 ? `${parts.slice(1).join(",").trim()} ${parts[0].trim()}` : name;
}

export function normalizeSearch(value) {
  return value.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase().trim();
}

export function filterRows(rows, state) {
  const query = normalizeSearch(state.query);
  return rows.filter(row => {
    if (state.date && row.forecast_date !== state.date) return false;
    if (query && !normalizeSearch(displayName(row.name)).includes(query)) return false;
    if (state.role !== "all" && (row.start_share >= 0.5 ? "starter" : "reliever") !== state.role) return false;
    if (state.view === "watchlist" && !row.performance_watchlist) return false;
    if (state.view === "alerts" && !row.alert) return false;
    if (state.view === "misses" && row.y_bf < 50) return false;
    return true;
  }).sort((a, b) => {
    const av = a[state.sort], bv = b[state.sort];
    if (av == null && bv != null) return 1;
    if (bv == null && av != null) return -1;
    const difference = av == null ? 0 : typeof av === "string" ? av.localeCompare(bv) : av - bv;
    return difference * state.direction || a.forecast_date.localeCompare(b.forecast_date) || a.pitcher - b.pitcher;
  });
}

export function historyThrough(rows, selected) {
  return rows.filter(row => row.pitcher === selected.pitcher && row.forecast_date <= selected.forecast_date)
    .sort((a, b) => a.forecast_date.localeCompare(b.forecast_date));
}

export function exportCsv(rows, reveal) {
  const fields = ["name", "forecast_date", "team", "performance_rank", "p_k_A", "p_bb_A",
    "p_k_B", "p_bb_B", "p_k_C", "p_bb_C", "change_delta", "repertoire_fallback", "alert",
    "n_pitches_cur", "chg_base_pitches", "fb_velo_cur", "chg_fb_velo", "fb_ivb_cur", "chg_fb_ivb",
    "fb_hb_cur", "chg_fb_hb", "use_fb_cur", "use_cut_cur", "use_brk_cur", "use_off_cur", "use_other_cur"];
  if (reveal) fields.push("y_bf", "y_k", "y_bb", "observed_k", "observed_bb", "observed_k_minus_bb", "resolution");
  const quote = value => {
    if (value == null) return "";
    let text = String(value);
    if (typeof value === "string" && /^[=+\-@]/.test(text)) text = `'${text}`;
    return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
  };
  return [fields.join(","), ...rows.map(row => fields.map(field => quote(row[field])).join(","))].join("\r\n");
}

export function readState(search) {
  const p = new URLSearchParams(search);
  const view = VIEWS.includes(p.get("view")) ? p.get("view") : "watchlist";
  return {
    season: p.get("season") === "2024" ? "2024" : "2025",
    date: p.get("date") ?? (["alerts", "misses"].includes(view) ? "all" : null),
    pitcher: Number(p.get("pitcher")) || null, at: p.get("at"),
    view, reveal: p.get("outcomes") === "1" || view === "misses", query: p.get("q") || "",
    role: ["starter", "reliever"].includes(p.get("role")) ? p.get("role") : "all",
    sort: view === "misses" ? "error_pp" : view === "alerts" ? "forecast_date" : "performance_rank",
    direction: view === "misses" ? -1 : 1, page: 0,
  };
}
