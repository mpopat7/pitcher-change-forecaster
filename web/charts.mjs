const percent = value => `${(value * 100).toFixed(0)}%`;
const shortDate = value => new Date(`${value}T12:00:00`).toLocaleDateString("en-US", { month: "short", day: "numeric" });

export const HISTORY_METRICS = {
  forecast: { field:"k_minus_bb_B", label:"Projected K% minus BB%", unit:"%", step:0.05 },
  velocity: { field:"fb_velo_cur", label:"Fastball velocity", unit:"mph", step:1 },
  movement: { field:"fb_ivb_cur", label:"Fastball vertical movement", unit:"in", step:2 },
  usage: { field:"use_fb_cur", label:"Fastball usage", unit:"%", step:0.1 },
};

export function chartValue(value, metric) {
  if (value == null) return "—";
  const definition = HISTORY_METRICS[metric];
  return definition.unit === "%" ? `${(value * 100).toFixed(1)}%` : `${value.toFixed(1)} ${definition.unit}`;
}

export function forecastChart(rows, reveal, metric = "forecast") {
  const definition = HISTORY_METRICS[metric];
  const field = definition.field;
  const showObserved = reveal && metric === "forecast";
  const width = 560, height = 200, left = 44, right = width - 12, top = 18, bottom = height - 32;
  const values = rows.flatMap(row => [row[field], ...(showObserved ? [row.observed_k_minus_bb] : [])]).filter(value => value != null);
  if (!values.length) return '<p class="minor-note">No measurements available through this forecast date.</p>';
  const step = definition.step;
  const low = Math.floor((Math.min(...values) - step / 2) / step) * step;
  const high = Math.ceil((Math.max(...values) + step / 2) / step) * step;
  const first = Date.parse(rows[0].forecast_date), last = Date.parse(rows.at(-1).forecast_date);
  const x = row => first === last ? (left + right) / 2 : left + (Date.parse(row.forecast_date) - first) / (last - first) * (right - left);
  const y = value => bottom - (value - low) / (high - low) * (bottom - top);
  const line = (key, className) => {
    let drawing = "", gap = true;
    for (const row of rows) {
      if (row[key] == null) { gap = true; continue; }
      drawing += `${gap ? "M" : "L"}${x(row).toFixed(2)},${y(row[key]).toFixed(2)} `;
      gap = false;
    }
    const dots = rows.filter(row => row[key] != null).map(row => `<circle cx="${x(row)}" cy="${y(row[key])}" r="3" class="${className}"><title>${shortDate(row.forecast_date)}: ${chartValue(row[key],metric)}</title></circle>`).join("");
    return `<path d="${drawing}" class="${className} chart-line"/>${dots}`;
  };
  const tickLabel = value => definition.unit === "%" ? percent(value) : value.toFixed(0);
  const ticks = [low, (low + high) / 2, high].map(value => `<line x1="${left}" x2="${right}" y1="${y(value)}" y2="${y(value)}" class="grid-line"/><text x="${left - 9}" y="${y(value) + 4}" text-anchor="end">${tickLabel(value)}</text>`).join("");
  return `<svg class="history-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${definition.label} (${definition.unit}) through ${rows.at(-1).forecast_date}. ${showObserved ? "Dashed line shows observed following-month rates." : "Only forecast-date information is shown."}">${ticks}${line(field, "forecast-series")}${showObserved ? line("observed_k_minus_bb", "observed-series") : ""}<text x="${left}" y="${height - 5}">${shortDate(rows[0].forecast_date)}</text>${rows.length > 1 ? `<text x="${right}" y="${height - 5}" text-anchor="end">${shortDate(rows.at(-1).forecast_date)}</text>` : ""}</svg>`;
}
