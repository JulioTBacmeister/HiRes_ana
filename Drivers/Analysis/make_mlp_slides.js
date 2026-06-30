/**
 * make_mlp_slides.js
 * ==================
 * Two modes:
 *   single — 2 or 3 slides for one MLP run
 *   sweep  — summary slide + 3-slide block per run
 */

const pptxgen = require("pptxgenjs");
const fs      = require("fs");

const metaPath = process.argv[2];
if (!metaPath) { console.error("Usage: node make_mlp_slides.js <meta.json>"); process.exit(1); }
const meta = JSON.parse(fs.readFileSync(metaPath, "utf8"));

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
  green:   "3C6E47",
};

const W = 13.3, H = 7.5;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function scoreColor(r2) {
  return r2 >= 0.5 ? C.good : r2 >= 0.3 ? C.warn : "C00000";
}
function makeShadow() {
  return { type:"outer", blur:5, offset:2, angle:135, color:"000000", opacity:0.12 };
}
function statBox(slide, x, y, w, h, label, value, valueColor) {
  slide.addShape("rect", { x, y, w, h,
    fill:{ color:C.white }, shadow:makeShadow(), line:{ color:C.light, width:0.5 } });
  slide.addText(label, { x:x+0.08, y:y+0.06, w:w-0.16, h:0.28,
    fontSize:9, color:C.muted, align:"center", fontFace:"Calibri", margin:0 });
  slide.addText(value, { x:x+0.06, y:y+0.32, w:w-0.12, h:0.38,
    fontSize:18, color:valueColor||C.dark, bold:true,
    align:"center", fontFace:"Calibri", margin:0 });
}
function addHeader(slide, title, accentColor) {
  const col = accentColor || C.dark;
  slide.addShape("rect", { x:0, y:0, w:W, h:0.70,
    fill:{ color:col }, line:{ color:col } });
  slide.addText(title, { x:0.35, y:0, w:9, h:0.70,
    fontSize:20, bold:true, color:C.white,
    fontFace:"Calibri", valign:"middle", margin:0 });
  slide.addText(meta.case, { x:9.0, y:0, w:4.0, h:0.70,
    fontSize:11, color:C.light, italic:true,
    fontFace:"Calibri", align:"right", valign:"middle", margin:0 });
}
function addPageNum(slide, n, total) {
  slide.addText(`${n} / ${total}`, { x:W-0.70, y:H-0.30, w:0.55, h:0.25,
    fontSize:8, color:C.muted, align:"right", fontFace:"Calibri", margin:0 });
}
function addFigureCard(slide, figPath, x, y, w, h) {
  slide.addShape("rect", { x:x-0.02, y:y-0.04, w:w+0.04, h:h+0.08,
    fill:{ color:C.white }, line:{ color:C.light, width:0.5 }, shadow:makeShadow() });
  slide.addImage({ path:figPath, x, y, w, h,
    sizing:{ type:"contain", w, h } });
}

const cardY=0.80, cardH=0.82, cardW=1.55, gap=0.18;

function addSHCards(slide, m) {
  const cards = [
    ["Train R²", m.train_r2.toFixed(3), scoreColor(m.train_r2)],
    ["Test R²",  m.test_r2.toFixed(3),  scoreColor(m.test_r2)],
    ["Test r",   m.test_r.toFixed(3),   scoreColor(m.test_r)],
    ["Train N",  m.train_n.toLocaleString(), C.dark],
    ["Test N",   m.test_n.toLocaleString(),  C.dark],
  ];
  cards.forEach(([lbl,val,col],i) =>
    statBox(slide, 0.30+i*(cardW+gap), cardY, cardW, cardH, lbl, val, col));
}

function addTransferCards(slide, m) {
  const cards = [
    ["Train R² (SH)", m.train_r2.toFixed(3),      scoreColor(m.train_r2)],
    ["Test R² (SH)",  m.test_r2.toFixed(3),        scoreColor(m.test_r2)],
    ["Test r (SH)",   m.test_r.toFixed(3),         scoreColor(m.test_r)],
    ["Transfer R²",   m.transfer_r2.toFixed(3),    scoreColor(m.transfer_r2)],
    ["Transfer r",    m.transfer_r.toFixed(3),     scoreColor(m.transfer_r)],
    ["Transfer N",    m.transfer_n.toLocaleString(), C.dark],
  ];
  const cw=1.38, g=0.16;
  cards.forEach(([lbl,val,col],i) =>
    statBox(slide, 0.28+i*(cw+g), cardY, cw, cardH, lbl, val, col));
}

// ---------------------------------------------------------------------------
// Shared slide builders
// ---------------------------------------------------------------------------
function buildRunSlides(pres, runMeta, slideNum, totalSlides) {
  const hasTransfer = runMeta.fig3 !== null && runMeta.fig3 !== undefined;

  // slide 1 — config + distributions
  const s1 = pres.addSlide();
  s1.background = { color: C.offwhite };
  addHeader(s1, `${runMeta.run_name || runMeta.case}  —  Config & Distributions`);
  addSHCards(s1, runMeta);

  // config table
  const cfgX=0.30, cfgY=1.75, cfgW=9.0, cfgH=1.20;
  s1.addShape("rect", { x:cfgX, y:cfgY, w:cfgW, h:cfgH,
    fill:{ color:C.white }, line:{ color:C.light, width:0.5 }, shadow:makeShadow() });
  const cfgRows = [
    ["Fields",     runMeta.fields],
    ["Predictors", `${runMeta.n_pred}  (${runMeta.hidden_dims} → 1)`],
    ["Loss / domain", `|error|^${runMeta.loss_power}  |  z: ${runMeta.z_targ}  |  ${runMeta.lat_lon}`],
  ];
  cfgRows.forEach(([k,v],i) => {
    const ry = cfgY + 0.10 + i*0.33;
    s1.addText(k+":", { x:cfgX+0.15, y:ry, w:1.50, h:0.28,
      fontSize:10, bold:true, color:C.mid, fontFace:"Calibri", margin:0 });
    s1.addText(v, { x:cfgX+1.70, y:ry, w:cfgW-1.85, h:0.28,
      fontSize:10, color:C.text, fontFace:"Calibri Light", margin:0 });
  });
  addFigureCard(s1, runMeta.fig1, 0.30, 3.05, W-0.60, 3.55);
  addPageNum(s1, slideNum, totalSlides);

  // slide 2 — predicted vs actual + importance
  const s2 = pres.addSlide();
  s2.background = { color: C.offwhite };
  addHeader(s2, `${runMeta.run_name || runMeta.case}  —  Prediction & Importance`);
  addSHCards(s2, runMeta);
  addFigureCard(s2, runMeta.fig2, 0.30, 1.85, W-0.60, 5.20);
  addPageNum(s2, slideNum+1, totalSlides);

  // slide 3 — transfer (optional)
  if (hasTransfer) {
    const s3 = pres.addSlide();
    s3.background = { color: C.offwhite };
    addHeader(s3, `${runMeta.run_name || runMeta.case}  —  Transfer: ${runMeta.transfer_label}`, C.green);
    addTransferCards(s3, runMeta);
    s3.addText(
      `Model trained on SH  →  applied to: ${runMeta.transfer_label}  (no retraining)`,
      { x:0.30, y:1.72, w:W-0.60, h:0.25,
        fontSize:10, italic:true, color:C.muted, fontFace:"Calibri", margin:0 }
    );
    addFigureCard(s3, runMeta.fig3, 0.30, 2.05, W-0.60, 5.00);
    addPageNum(s3, slideNum+2, totalSlides);
    return 3;
  }
  return 2;
}

// ---------------------------------------------------------------------------
// Overview slide builder
// ---------------------------------------------------------------------------
function buildOverviewSlide(pres, m, slideNum, totalSlides) {
  const s = pres.addSlide();
  s.background = { color: C.offwhite };

  // header
  addHeader(s, "MLP Model Overview");

  // ── column layout ─────────────────────────────────────────────────────────
  // Three equal columns, y starts below header + small margin
  const colY  = 0.85;
  const colH  = H - colY - 0.40;
  const colW  = 3.95;
  const colX  = [0.28, 4.50, 8.72];
  const padX  = 0.18;
  const headH = 0.40;

  function colCard(slide, ci, title, accentCol) {
    // card background
    slide.addShape("rect", { x:colX[ci], y:colY, w:colW, h:colH,
      fill:{ color:C.white }, line:{ color:C.light, width:0.8 },
      shadow:makeShadow() });
    // coloured title bar
    slide.addShape("rect", { x:colX[ci], y:colY, w:colW, h:headH,
      fill:{ color:accentCol }, line:{ color:accentCol } });
    slide.addText(title, { x:colX[ci]+padX, y:colY, w:colW-padX*2, h:headH,
      fontSize:13, bold:true, color:C.white,
      fontFace:"Calibri", valign:"middle", margin:0 });
  }

  function bullet(slide, ci, items, startY) {
    // cumulative Y so mixed parent/sub rows don't overlap
    let yy = startY;
    items.forEach((item) => {
      const indent = item.sub ? 0.30 : 0.08;
      const symbol = item.sub ? "-" : ">";
      const fs     = item.sub ? 9.0 : 10.0;
      const color  = item.sub ? C.muted : C.text;
      const lh     = item.sub ? 0.30 : 0.34;   // line height
      const gap    = item.sub ? 0.01 : 0.06;   // gap before item
      yy += gap;
      slide.addText(`${symbol}  ${item.text}`, {
        x: colX[ci] + padX + indent, y: yy,
        w: colW - padX*2 - indent,   h: lh,
        fontSize: fs, color: color, fontFace: "Calibri", margin: 0,
        bold: item.bold || false,
      });
      yy += lh;
    });
  }

  // ── COLUMN 1: Architecture ─────────────────────────────────────────────
  colCard(s, 0, "Architecture", C.dark);

  // mini network diagram — text boxes with arrows
  const archX = colX[0] + 0.25;
  const layers = [
    { label: `Input  (${m.n_pred})`,    col: "4472C4" },
    { label: "256 neurons",             col: "1C7293" },
    { label: "256 neurons",             col: "1C7293" },
    { label: "256 neurons",             col: "1C7293" },
    { label: "128 neurons",             col: "2E75B6" },
    { label: "Output  (1)",             col: "C00000" },
  ];
  const boxW=3.45, boxH=0.30, boxGap=0.10;
  const archStartY = colY + headH + 0.12;
  layers.forEach((l, i) => {
    const by = archStartY + i*(boxH+boxGap);
    s.addShape("rect", { x:archX, y:by, w:boxW, h:boxH,
      fill:{ color:l.col }, line:{ color:l.col },
      rectRadius: 0.04 });
    s.addText(l.label, { x:archX, y:by, w:boxW, h:boxH,
      fontSize:9.5, color:C.white, bold:true,
      align:"center", valign:"middle", fontFace:"Calibri", margin:0 });
    // arrow between layers
    if (i < layers.length-1) {
      const ay = by + boxH + 0.01;
      s.addText("↓", { x:archX + boxW/2 - 0.15, y:ay, w:0.30, h:boxGap+0.02,
        fontSize:9, color:C.muted, align:"center", fontFace:"Calibri", margin:0 });
    }
  });

  // per-layer ops note
  const opsY = archStartY + layers.length*(boxH+boxGap) + 0.05;
  s.addText("Each hidden layer:  Linear → BatchNorm → LeakyReLU(0.1) → Dropout(0.2)",
    { x:archX-0.05, y:opsY, w:boxW+0.10, h:0.38,
      fontSize:8.5, color:C.muted, italic:true,
      fontFace:"Calibri", align:"center", margin:0 });

  // ── COLUMN 2: Training setup ──────────────────────────────────────────
  colCard(s, 1, "Training Setup", "2E75B6");

  const trainItems = [
    { text:"Target transform" },
    { text:"log(epwp)  →  StandardScaler", sub:true },
    { text:"Back-transform:  exp( · )", sub:true },
    { text:"Loss function" },
    { text:`|error|^${m.loss_power}  — emphasises strong events`, sub:true },
    { text:"Optimizer" },
    { text:"AdamW  (weight decay 1e-5)", sub:true },
    { text:"ReduceLROnPlateau scheduler", sub:true },
    { text:"Early stopping on val loss", sub:true },
    { text:"Train / test split" },
    { text:"Chronological — no random shuffle", sub:true },
    { text:"Train: t=[48,248)   Test: t=[0,28)", sub:true },
    { text:"Val: last 10% of training block", sub:true },
  ];
  bullet(s, 1, trainItems, colY + headH + 0.18);

  // ── COLUMN 3: Prediction context ─────────────────────────────────────
  colCard(s, 2, "Prediction Context", C.green);

  const predItems = [
    { text:"Inputs (predictors)" },
    { text:`${m.n_pred} features from tropospheric fields`, sub:true },
    { text:`u, v, stab, zeta, tilt, fgf`, sub:true },
    { text:"At 7 vertical levels, 4 timesteps", sub:true },
    { text:"Spatial patch: " + m.lat_lon, sub:true },
    { text:"Target" },
    { text:`epwp (GW momentum flux) at ${m.z_targ}`, sub:true },
    { text:"Pressure-weighted, spatially averaged", sub:true },
    { text:"Training domain" },
    { text:"Southern Ocean (SH), DYAMOND 3.75km", sub:true },
    { text:"40-day record, Aug 2016", sub:true },
    { text:"Transfer test" },
    { text:"SH-trained model → NH (no retraining)", sub:true },
    { text:"Tests physical generality of predictors", sub:true },
  ];
  bullet(s, 2, predItems, colY + headH + 0.18);

  addPageNum(s, slideNum, totalSlides);
}

// ---------------------------------------------------------------------------
// MAIN
// ---------------------------------------------------------------------------
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "mlp_to_pptx.py";
pres.title  = `MLP Results -- ${meta.case}`;

if (meta.mode === "sweep") {
  const hasTransfer = meta.runs[0].fig3 !== null && meta.runs[0].fig3 !== undefined;
  const slidesPerRun = hasTransfer ? 3 : 2;
  const totalSlides  = 2 + meta.runs.length * slidesPerRun;

  // slide 1: overview
  buildOverviewSlide(pres, meta.runs[0], 1, totalSlides);

  // slide 2: summary
  const ss = pres.addSlide();
  ss.background = { color: C.offwhite };
  addHeader(ss, "Predictor Sweep Summary");

  const tbl = meta.sweep_summary;
  const colW = [2.2, 1.4, 1.4, 1.4, 1.4];
  const hdrs = ["Run", "Train R2", "Test R2", "Test r", "Transfer r"];
  const tblX = 0.30, tblY = 0.85, rowH = 0.32;

  hdrs.forEach((h, ci) => {
    const cx = tblX + colW.slice(0,ci).reduce((a,b)=>a+b,0);
    ss.addShape("rect", { x:cx, y:tblY, w:colW[ci], h:rowH,
      fill:{ color:C.dark }, line:{ color:C.dark } });
    ss.addText(h, { x:cx+0.05, y:tblY, w:colW[ci]-0.05, h:rowH,
      fontSize:10, bold:true, color:C.white, valign:"middle",
      fontFace:"Calibri", margin:0 });
  });

  tbl.forEach((row, ri) => {
    const ry  = tblY + rowH*(ri+1);
    const bg  = ri % 2 === 0 ? "EBF3FB" : C.white;
    const vals = [
      row.run_name,
      row.train_r2.toFixed(3),
      row.test_r2.toFixed(3),
      row.test_r.toFixed(3),
      row.transfer_r !== null ? row.transfer_r.toFixed(3) : "--",
    ];
    vals.forEach((v, ci) => {
      const cx = tblX + colW.slice(0,ci).reduce((a,b)=>a+b,0);
      ss.addShape("rect", { x:cx, y:ry, w:colW[ci], h:rowH,
        fill:{ color:bg }, line:{ color:C.light, width:0.3 } });
      const col = ci === 0 ? C.text :
                  ci === 2 ? scoreColor(row.test_r2) :
                  ci === 3 ? scoreColor(row.test_r)  :
                  ci === 4 && row.transfer_r !== null ? scoreColor(row.transfer_r) : C.text;
      ss.addText(v, { x:cx+0.05, y:ry, w:colW[ci]-0.05, h:rowH,
        fontSize:10, color:col, bold:(ci>0), valign:"middle",
        fontFace:"Calibri", margin:0 });
    });
  });

  const figH = H - tblY - rowH*(tbl.length+1) - 0.55;
  const figY = tblY + rowH*(tbl.length+1) + 0.15;
  addFigureCard(ss, meta.fig_summary, 0.30, figY, W-0.60, figH);
  addPageNum(ss, 2, totalSlides);

  // per-run slides starting at slide 3
  let slideNum = 3;
  meta.runs.forEach(runMeta => {
    slideNum += buildRunSlides(pres, runMeta, slideNum, totalSlides);
  });

} else {
  // single mode
  const hasTransfer = meta.fig3 !== null && meta.fig3 !== undefined;
  const totalSlides = hasTransfer ? 4 : 3;
  buildOverviewSlide(pres, meta, 1, totalSlides);
  buildRunSlides(pres, meta, 2, totalSlides);
}

pres.writeFile({ fileName: meta.pptx_out })
  .then(() => console.log(`PPTX written -> ${meta.pptx_out}`))
  .catch(err => { console.error(err); process.exit(1); });
