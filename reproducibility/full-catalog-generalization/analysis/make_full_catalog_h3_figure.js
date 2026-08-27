const fs = require("fs");
const path = require("path");
const sharp = require("sharp");

const root = path.resolve(__dirname, "..");
const inputPath = path.join(root, "results", "analysis", "full_catalog_event_deltas.csv");
const outputDir = path.join(root, "results", "figures");
fs.mkdirSync(outputDir, { recursive: true });

function parseSimpleCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const header = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const values = line.split(",");
    return Object.fromEntries(header.map((name, i) => [name, values[i]]));
  });
}

function escapeXml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

const rows = parseSimpleCsv(fs.readFileSync(inputPath, "utf8"))
  .map((row) => ({
    ...row,
    delta: Number(row.delta_chi_eff_minus_luminosity_distance),
    strict: row.strict_population_member === "True",
  }))
  .sort((a, b) => a.delta - b.delta);

const strict = rows.filter((row) => row.strict);
const positiveAll = rows.filter((row) => row.delta > 0).length;
const positiveStrict = strict.filter((row) => row.delta > 0).length;
const strictMedian = [...strict.map((row) => row.delta)].sort((a, b) => a - b)[
  Math.floor(strict.length / 2)
];

const width = 1700;
const height = 1050;
const margin = { left: 165, right: 75, top: 220, bottom: 135 };
const plotWidth = width - margin.left - margin.right;
const plotHeight = height - margin.top - margin.bottom;
const dataMin = Math.min(...rows.map((row) => row.delta));
const dataMax = Math.max(...rows.map((row) => row.delta));
const yMin = Math.min(-0.14, Math.floor((dataMin - 0.01) / 0.05) * 0.05);
const yMax = Math.max(0.2, Math.ceil((dataMax + 0.01) / 0.05) * 0.05);

const x = (index) => margin.left + (index / (rows.length - 1)) * plotWidth;
const y = (value) =>
  margin.top + ((yMax - value) / (yMax - yMin)) * plotHeight;

const yTicks = [];
for (let value = Math.ceil(yMin / 0.05) * 0.05; value <= yMax + 1e-9; value += 0.05) {
  yTicks.push(Number(value.toFixed(2)));
}

const parts = [];
parts.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`);
parts.push(`<rect width="100%" height="100%" fill="#ffffff"/>`);
parts.push(`<style>
  text { font-family: Arial, Helvetica, sans-serif; fill: #172033; }
  .title { font-size: 36px; font-weight: 700; }
  .subtitle { font-size: 23px; fill: #475569; }
  .axis { font-size: 22px; }
  .tick { font-size: 18px; fill: #475569; }
  .legend { font-size: 19px; }
  .annotation { font-size: 16px; font-weight: 600; fill: #991b1b; }
</style>`);
parts.push(`<text class="title" x="${margin.left}" y="58">Full-catalog parameter-disagreement ordering</text>`);
parts.push(`<text class="subtitle" x="${margin.left}" y="100">XPHM–XPNR; Δ NW1 = χ_eff NW1 − luminosity-distance NW1</text>`);
parts.push(`<text class="subtitle" x="${margin.left}" y="133">Post-lock generalization: ${positiveAll}/104 positive overall; ${positiveStrict}/93 positive among strict matched-settings pairs</text>`);

const legendY = 174;
parts.push(`<circle cx="${margin.left + 4}" cy="${legendY}" r="7" fill="#2563eb"/><text class="legend" x="${margin.left + 21}" y="${legendY + 7}">strict matched-settings pair</text>`);
parts.push(`<rect x="${margin.left + 432}" y="${legendY - 7}" width="14" height="14" fill="#d97706"/><text class="legend" x="${margin.left + 455}" y="${legendY + 7}">non-Tier-A pair (descriptive)</text>`);
parts.push(`<circle cx="${margin.left + 945}" cy="${legendY}" r="7" fill="#b91c1c"/><text class="legend" x="${margin.left + 962}" y="${legendY + 7}">negative Δ exception</text>`);

for (const tick of yTicks) {
  const yy = y(tick);
  const isZero = Math.abs(tick) < 1e-12;
  parts.push(`<line x1="${margin.left}" x2="${width - margin.right}" y1="${yy}" y2="${yy}" stroke="${isZero ? "#111827" : "#dbe3ed"}" stroke-width="${isZero ? 2.5 : 1}"/>`);
  parts.push(`<text class="tick" x="${margin.left - 18}" y="${yy + 6}" text-anchor="end">${tick.toFixed(2)}</text>`);
}

const medianY = y(strictMedian);
parts.push(`<line x1="${margin.left}" x2="${width - margin.right}" y1="${medianY}" y2="${medianY}" stroke="#0f766e" stroke-width="2" stroke-dasharray="10 8" opacity="0.8"/>`);
parts.push(`<text class="tick" x="${width - margin.right - 8}" y="${medianY - 9}" text-anchor="end" fill="#0f766e">strict median = ${strictMedian.toFixed(3)}</text>`);

rows.forEach((row, index) => {
  const xx = x(index);
  const yy = y(row.delta);
  const zeroY = y(0);
  const negative = row.delta < 0;
  const color = negative ? "#b91c1c" : row.strict ? "#2563eb" : "#d97706";
  parts.push(`<line x1="${xx}" x2="${xx}" y1="${zeroY}" y2="${yy}" stroke="${color}" stroke-width="2" opacity="0.48"/>`);
  if (row.strict) {
    parts.push(`<circle cx="${xx}" cy="${yy}" r="6" fill="${color}" stroke="#ffffff" stroke-width="1.5"/>`);
  } else {
    parts.push(`<rect x="${xx - 6}" y="${yy - 6}" width="12" height="12" rx="1.5" fill="${color}" stroke="#ffffff" stroke-width="1.5"/>`);
  }
});

const negativeRows = rows.filter((row) => row.delta < 0);
const calloutX = margin.left + 130;
const calloutY = y(-0.012);
const calloutWidth = 470;
const calloutHeight = 42 + negativeRows.length * 27;
parts.push(`<rect x="${calloutX}" y="${calloutY}" width="${calloutWidth}" height="${calloutHeight}" rx="9" fill="#fff7f7" stroke="#fecaca"/>`);
parts.push(`<text class="annotation" x="${calloutX + 16}" y="${calloutY + 27}">Five negative-Δ exceptions</text>`);
negativeRows.forEach((row, i) => {
  parts.push(`<text class="tick" x="${calloutX + 16}" y="${calloutY + 55 + i * 27}" fill="#7f1d1d">${escapeXml(row.event)}  (${row.delta.toFixed(3)})</text>`);
});

parts.push(`<line x1="${margin.left}" x2="${margin.left}" y1="${margin.top}" y2="${height - margin.bottom}" stroke="#334155" stroke-width="2"/>`);
parts.push(`<line x1="${margin.left}" x2="${width - margin.right}" y1="${height - margin.bottom}" y2="${height - margin.bottom}" stroke="#334155" stroke-width="2"/>`);
parts.push(`<text class="axis" x="${margin.left + plotWidth / 2}" y="${height - 45}" text-anchor="middle">Events sorted by Δ NW1</text>`);
parts.push(`<text class="axis" x="42" y="${margin.top + plotHeight / 2}" text-anchor="middle" transform="rotate(-90 42 ${margin.top + plotHeight / 2})">Δ NW1 (χ_eff − luminosity distance)</text>`);
parts.push(`<text class="tick" x="${margin.left}" y="${height - margin.bottom + 32}" text-anchor="middle">1</text>`);
parts.push(`<text class="tick" x="${width - margin.right}" y="${height - margin.bottom + 32}" text-anchor="middle">104</text>`);

parts.push(`</svg>`);
const svg = parts.join("\n");
const svgPath = path.join(outputDir, "figure_full_catalog_h3_delta_sorted.svg");
const pngPath = path.join(outputDir, "figure_full_catalog_h3_delta_sorted.png");
fs.writeFileSync(svgPath, svg, "utf8");

sharp(Buffer.from(svg))
  .png({ compressionLevel: 9 })
  .toFile(pngPath)
  .then(() => {
    console.log(JSON.stringify({ svg: svgPath, png: pngPath, events: rows.length }, null, 2));
  });
