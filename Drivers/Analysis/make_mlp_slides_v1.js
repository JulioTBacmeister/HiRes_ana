/**
 * make_mlp_slides.js
 * ==================
 * Assemble a 2- or 3-slide PowerPoint from pre-rendered PNG figures + metadata.
 *
 * Slide 1 — Run config + distribution diagnostics (SH test)
 * Slide 2 — Predicted vs actual + feature importance (SH test)
 * Slide 3 — Transfer diagnostics (NH, optional — only if meta.fig3 is set)
 */

const pptxgen = require("pptxgenjs");
const fs      = require("fs");

const metaPath = process.argv[2];
if (!metaPath) { console.error("Usage: node make_mlp_slides.js <meta.json>"); process.exit(1); }
const meta = JSON.parse(fs.readFileSync(metaPath, "utf8"));

const hasTransfer = meta.fig3 !== null && meta.fig3 !== undefined;
const nSlides     = hasTransfer ? 3 : 2;

// ---------------------------------------------------------------------------
// Colour palette
// ---------------------------------------------------------------------------
const C = {
  dark:    "065A82",
  mid:     "1C7293",
  light:   "9DCDE4",
  white:   "FFFFFF",
  offwhite:"F4F8FB",
  text:    "1A1A2E",
  muted:   "4A6274",
  good:    "1D7A4E",
  warn:    "C05000",
  green:   "375623",   // transfer accent
};

const W = 13.3, H = 7.5;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function scoreColor(r2) {
  return r2 >= 0.5 ? C.good : r2 >= 0.3 ? C.warn : "C00000";
}

function makeShadow() {
  return { type: "outer", blur: 5, offset: 2, angle: 135, color: "000000", opacity: 0.12 };
}

function statBox(slide, x, y, w, h, label, value, valueColor) {
  slide.addShape("rect", {
    x, y, w, h,
    fill: { color: C.white },
    shadow: makeShadow(),
    line: { color: C.light, width: 0.5 },
  });
  slide.addText(label, {
    x: x + 0.08, y: y + 0.06, w: w - 0.16, h: 0.28,
    fontSize: 9, color: C.muted, align: "center",
    fontFace: "Calibri", margin: 0,
  });
  slide.addText(value, {
    x: x + 0.06, y: y + 0.32, w: w - 0.12, h: 0.38,
    fontSize: 18, color: valueColor || C.dark, bold: true,
    align: "center", fontFace: "Calibri", margin: 0,
  });
}

function addHeader(slide, title, accentColor) {
  const col = accentColor || C.dark;
  slide.addShape("rect", { x: 0, y: 0, w: W, h: 0.70,
    fill: { color: col }, line: { color: col } });
  slide.addText(title, {
    x: 0.35, y: 0, w: 9, h: 0.70,
    fontSize: 22, bold: true, color: C.white,
    fontFace: "Calibri", valign: "middle", margin: 0,
  });
  slide.addText(meta.case, {
    x: 9.0, y: 0, w: 4.0, h: 0.70,
    fontSize: 12, color: C.light, italic: true,
    fontFace: "Calibri", align: "right", valign: "middle", margin: 0,
  });
}

function addPageNum(slide, n) {
  slide.addText(`${n} / ${nSlides}`, {
    x: W - 0.70, y: H - 0.30, w: 0.55, h: 0.25,
    fontSize: 8, color: C.muted, align: "right", fontFace: "Calibri", margin: 0,
  });
}

function addFigureCard(slide, figPath, x, y, w, h) {
  slide.addShape("rect", {
    x: x - 0.02, y: y - 0.04, w: w + 0.04, h: h + 0.08,
    fill: { color: C.white }, line: { color: C.light, width: 0.5 },
    shadow: makeShadow(),
  });
  slide.addImage({ path: figPath, x, y, w, h,
    sizing: { type: "contain", w, h } });
}

// ---------------------------------------------------------------------------
// Shared card layout
// ---------------------------------------------------------------------------
const cardY = 0.80, cardH = 0.82, cardW = 1.55, gap = 0.18;

function addSHCards(slide) {
  const cards = [
    ["Train R²", meta.train_r2.toFixed(3), scoreColor(meta.train_r2)],
    ["Test R²",  meta.test_r2.toFixed(3),  scoreColor(meta.test_r2)],
    ["Test r",   meta.test_r.toFixed(3),   scoreColor(meta.test_r)],
    ["Train N",  meta.train_n.toLocaleString(), C.dark],
    ["Test N",   meta.test_n.toLocaleString(),  C.dark],
  ];
  cards.forEach(([lbl, val, col], i) => {
    statBox(slide, 0.30 + i * (cardW + gap), cardY, cardW, cardH, lbl, val, col);
  });
}

function addTransferCards(slide) {
  const cards = [
    ["Train R² (SH)",    meta.train_r2.toFixed(3),        scoreColor(meta.train_r2)],
    ["Test R² (SH)",     meta.test_r2.toFixed(3),         scoreColor(meta.test_r2)],
    ["Test r (SH)",      meta.test_r.toFixed(3),          scoreColor(meta.test_r)],
    ["Transfer R²",      meta.transfer_r2.toFixed(3),     scoreColor(meta.transfer_r2)],
    ["Transfer r",       meta.transfer_r.toFixed(3),      scoreColor(meta.transfer_r)],
    ["Transfer N",       meta.transfer_n.toLocaleString(), C.dark],
  ];
  // 6 cards — tighten spacing slightly
  const cw = 1.38, g = 0.16;
  cards.forEach(([lbl, val, col], i) => {
    statBox(slide, 0.28 + i * (cw + g), cardY, cw, cardH, lbl, val, col);
  });
}

// ---------------------------------------------------------------------------
// Presentation
// ---------------------------------------------------------------------------
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "mlp_to_pptx.py";
pres.title  = `MLP Results — ${meta.case}`;

// ===========================================================================
// SLIDE 1 — Config + distribution diagnostics
// ===========================================================================
const s1 = pres.addSlide();
s1.background = { color: C.offwhite };
addHeader(s1, "MLP Training Results");
addSHCards(s1);

// config table
const cfgX = 0.30, cfgY = 1.75, cfgW = 9.0, cfgH = 1.20;
s1.addShape("rect", { x: cfgX, y: cfgY, w: cfgW, h: cfgH,
  fill: { color: C.white }, line: { color: C.light, width: 0.5 },
  shadow: makeShadow() });
const cfgRows = [
  ["Fields",     meta.fields],
  ["Predictors", `${meta.n_pred}  (${meta.hidden_dims} → 1)`],
  ["Loss power", `|error|^${meta.loss_power}   |   z target: ${meta.z_targ}   |   domain: ${meta.lat_lon}`],
];
cfgRows.forEach(([k, v], i) => {
  const rowY = cfgY + 0.10 + i * 0.33;
  s1.addText(k + ":", { x: cfgX + 0.15, y: rowY, w: 1.30, h: 0.28,
    fontSize: 10, bold: true, color: C.mid, fontFace: "Calibri", margin: 0 });
  s1.addText(v, { x: cfgX + 1.50, y: rowY, w: cfgW - 1.70, h: 0.28,
    fontSize: 10, color: C.text, fontFace: "Calibri Light", margin: 0 });
});

addFigureCard(s1, meta.fig1, 0.30, 3.05, W - 0.60, 3.55);
addPageNum(s1, 1);

// ===========================================================================
// SLIDE 2 — Predicted vs actual + feature importance
// ===========================================================================
const s2 = pres.addSlide();
s2.background = { color: C.offwhite };
addHeader(s2, "Prediction Quality & Feature Importance");
addSHCards(s2);
addFigureCard(s2, meta.fig2, 0.30, 1.85, W - 0.60, 5.20);
addPageNum(s2, 2);

// ===========================================================================
// SLIDE 3 — Transfer diagnostics (optional)
// ===========================================================================
if (hasTransfer) {
  const s3 = pres.addSlide();
  s3.background = { color: C.offwhite };

  // green-tinted header for visual distinction
  const transferHeader = "3C6E47";
  addHeader(s3, `Transfer: ${meta.transfer_label}`, transferHeader);
  addTransferCards(s3);

  // label strip below cards
  s3.addText(
    `Model trained on SH  →  applied to: ${meta.transfer_label}  (no retraining)`,
    { x: 0.30, y: 1.72, w: W - 0.60, h: 0.25,
      fontSize: 10, italic: true, color: C.muted,
      fontFace: "Calibri", margin: 0 }
  );

  addFigureCard(s3, meta.fig3, 0.30, 2.05, W - 0.60, 5.00);
  addPageNum(s3, 3);
}

// ===========================================================================
// Write
// ===========================================================================
pres.writeFile({ fileName: meta.pptx_out })
  .then(() => console.log(`PPTX written → ${meta.pptx_out}`))
  .catch(err => { console.error(err); process.exit(1); });
