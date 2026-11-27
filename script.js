/* ===== BananaCheck — script.js =====
   Bagian 1: navbar   Bagian 2: pemuat model 3D   Bagian 3: area deteksi
   Deteksi YOLO26 berjalan di browser (onnxruntime-web, WebGPU/WASM). Model 3D = visual pendamping. */
const $ = (id) => document.getElementById(id);

/* ---------- 1. NAVBAR ---------- */
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
const burger = $('burger'), menu = $('menu');
function setMenu(open) {
  menu.classList.toggle('open', open);
  burger.setAttribute('aria-expanded', String(open));
  burger.setAttribute('aria-label', open ? 'Tutup menu' : 'Buka menu');
}
burger.addEventListener('click', () => setMenu(!menu.classList.contains('open')));
menu.addEventListener('click', (e) => { if (e.target.closest('a')) setMenu(false); });
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') setMenu(false); });
$('year').textContent = new Date().getFullYear();

/* ---------- 2. MODEL 3D ---------- */
// Dimuat terpisah: jika Three.js gagal diunduh, pesan error tampil di kotak model
// dan fitur lain (navbar, upload, deteksi) tetap jalan.
function showViewerError(text) {
  document.querySelectorAll('.v-status').forEach((el) => { el.textContent = text; el.classList.add('error'); el.hidden = false; });
}
let companion = null, pendingModel = null;
const viewerReady = import('./viewer3d.js')
  .then((m) => {
    m.initViewers();
    companion = m.initCompanion($('companion'));
    if (companion && pendingModel) companion.setModel(pendingModel);
  })
  .catch((err) => {
    console.error(err);
    showViewerError('Three.js gagal dimuat (' + (err && err.message ? err.message : err) + '). Periksa koneksi internet, lalu muat ulang halaman.');
  });
setTimeout(() => {
  document.querySelectorAll('.viewer[data-model] .v-status:not(.error)').forEach((el) => {
    if (!el.hidden) el.textContent += ' (masih berjalan — koneksi mungkin lambat)';
  });
}, 20000);

/* ---------- 3. DETEKSI DI BROWSER (ONNX Runtime Web) ---------- */
const DEFAULT_MODEL = 'models/best.onnx';
const ORT_BASE = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
const CLASS_NAMES = ['ripe', 'unripe'];            // urutan kelas saat training
const MODEL_3D = { RIPE: 'banana_ripe.glb', UNRIPE: 'banana_unripe.glb' };
const IOU_THRESHOLD = 0.45;
const MAX_BYTES = 5 * 1024 * 1024;
const ALLOWED = ['image/jpeg', 'image/png'];
let SIZE = 640;                                    // dibaca ulang dari model jika tersedia

const fileInput = $('file'), dropEmpty = $('dropEmpty'), stage = $('stage'), img = $('preview'), overlay = $('overlay'), video = $('video');
const detectBtn = $('detectBtn'), clearBtn = $('clearBtn'), loading = $('loading'), msg = $('msg'), preds = $('preds'), drop = $('drop');
const modelStatus = $('modelStatus'), modelFile = $('modelFile'), loadModelBtn = $('loadModelBtn'), defaultModelBtn = $('defaultModelBtn'), modelHint = $('modelHint');
const camSource = $('camSource'), startCamBtn = $('startCamBtn'), stopCamBtn = $('stopCamBtn');
const conf = $('conf'), confVal = $('confVal');
const mDet = $('mDet'), mInf = $('mInf'), mFps = $('mFps');
const companionLabel = $('companionLabel');

let currentFile = null, objectUrl = null;
let session = null, deviceLabel = '', modelBusy = false, modelName = '';
let camStream = null, camActive = false, lastStatic = null;

function setMsg(text, isError = false) { msg.textContent = text; msg.classList.toggle('err', isError); }
function setStatus(text, state) { modelStatus.textContent = text; modelStatus.dataset.state = state; }
const threshold = () => parseFloat(conf.value) || 0.25;

/* ----- 3a. Memuat model ----- */
async function gpuName() {
  try {
    const ad = await navigator.gpu.requestAdapter();
    const info = ad && (ad.info || (ad.requestAdapterInfo && await ad.requestAdapterInfo()));
    return info ? [info.vendor, info.architecture || info.description].filter(Boolean).join(' ') : '';
  } catch { return ''; }
}

function newInput(size) {
  return new ort.Tensor('float32', new Float32Array(3 * size * size).fill(0.447), [1, 3, size, size]);
}

async function createSession(buffer) {
  const tries = [];
  if (navigator.gpu) tries.push({ ep: 'webgpu', label: 'WebGPU' });
  tries.push({ ep: 'wasm', label: 'WASM (CPU)' });
  const errs = [];
  for (const t of tries) {
    try {
      const s = await ort.InferenceSession.create(new Uint8Array(buffer), { executionProviders: [t.ep], graphOptimizationLevel: 'all' });
      const shape = s.inputMetadata && s.inputMetadata[0] && s.inputMetadata[0].shape;
      if (shape && Number.isInteger(shape[2]) && shape[2] > 0) SIZE = shape[2];
      await s.run({ [s.inputNames[0]]: newInput(SIZE) });     // pemanasan sekaligus uji backend
      let label = t.label;
      if (t.ep === 'webgpu') { const n = await gpuName(); if (n) label += ' · ' + n; }
      return { s, label };
    } catch (err) { console.warn('Backend ' + t.ep + ' gagal:', err); errs.push(t.ep + ': ' + (err && err.message || err)); }
  }
  throw new Error(errs.join(' | ') || 'Tidak ada backend yang bisa dipakai');
}

async function loadModel(useLocal) {
  if (modelBusy) return;
  if (!window.ort) { setStatus('onnxruntime-web gagal dimuat (cek koneksi internet)', 'error'); return; }
  modelBusy = true; loadModelBtn.disabled = defaultModelBtn.disabled = true;
  const local = useLocal && modelFile.files[0];
  setStatus('Loading model…', 'loading');
  try {
    ort.env.wasm.wasmPaths = ORT_BASE;
    ort.env.wasm.proxy = false;
    ort.env.wasm.numThreads = 1;      // GitHub Pages tidak cross-origin isolated
    let buffer, name;
    if (local) { buffer = await local.arrayBuffer(); name = local.name; }
    else {
      const res = await fetch(DEFAULT_MODEL);
      if (!res.ok) throw new Error(`${DEFAULT_MODEL} tidak ditemukan (HTTP ${res.status})`);
      buffer = await res.arrayBuffer(); name = DEFAULT_MODEL;
    }
    const old = session; session = null;
    if (old && old.release) { try { await old.release(); } catch {} }
    const { s, label } = await createSession(buffer);
    session = s; deviceLabel = label; modelName = name;
    setStatus(`Model ready · ${label}`, 'ready');
    modelHint.textContent = `Model aktif: ${name}. Jalan di ${label}.`;
    if (lastStatic) redrawStatic();
  } catch (err) {
    console.error(err);
    session = null;
    setStatus('Gagal memuat model: ' + (err.message || err), 'error');
    if (!local) modelHint.textContent = 'Pastikan models/best.onnx ada dan website dibuka lewat server (bukan file://), atau pilih file .onnx lokal.';
  } finally {
    modelBusy = false; loadModelBtn.disabled = defaultModelBtn.disabled = false;
  }
}
async function ensureModel() {
  while (modelBusy) await new Promise((r) => setTimeout(r, 100));
  if (!session) await loadModel(!!modelFile.files[0]);
  return !!session;
}

loadModelBtn.addEventListener('click', () => loadModel(true));
defaultModelBtn.addEventListener('click', () => { modelFile.value = ''; loadModel(false); });
modelFile.addEventListener('change', () => {
  if (modelFile.files[0]) setStatus(`Dipilih: ${modelFile.files[0].name} — tekan Load model`, 'waiting');
});
// Muat model default otomatis begitu bagian deteksi mendekati layar
new IntersectionObserver((entries, obs) => {
  if (entries.some((e) => e.isIntersecting)) { obs.disconnect(); if (!session && !modelBusy) loadModel(false); }
}, { rootMargin: '300px' }).observe($('deteksi'));

/* ----- 3b. Pra-proses, inferensi, pasca-proses ----- */
const work = document.createElement('canvas');
const wctx = work.getContext('2d', { willReadFrequently: true });
let inputBuf = null;

function preprocess(src, sw, sh) {
  if (work.width !== SIZE) { work.width = work.height = SIZE; inputBuf = null; }
  if (!inputBuf) inputBuf = new Float32Array(3 * SIZE * SIZE);
  const r = Math.min(SIZE / sw, SIZE / sh);                   // letterbox, rasio tetap
  const nw = Math.round(sw * r), nh = Math.round(sh * r);
  const px = Math.floor((SIZE - nw) / 2), py = Math.floor((SIZE - nh) / 2);
  wctx.fillStyle = 'rgb(114,114,114)'; wctx.fillRect(0, 0, SIZE, SIZE);
  wctx.drawImage(src, 0, 0, sw, sh, px, py, nw, nh);
  const d = wctx.getImageData(0, 0, SIZE, SIZE).data, n = SIZE * SIZE;
  for (let i = 0, j = 0; i < n; i++, j += 4) {
    inputBuf[i] = d[j] / 255; inputBuf[n + i] = d[j + 1] / 255; inputBuf[2 * n + i] = d[j + 2] / 255;
  }
  return { r, px, py };
}

function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1), x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return ua > 0 ? inter / ua : 0;
}
function nms(list) {
  list.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const d of list) if (keep.every((k) => k.cls !== d.cls || iou(k, d) < IOU_THRESHOLD)) keep.push(d);
  return keep;
}

// Mengubah keluaran mentah YOLO menjadi kotak pada koordinat gambar asli.
function decode(out, meta, minScore, sw, sh) {
  const [, d1, d2] = out.dims, data = out.data, list = [];
  const clamp = (v, m) => Math.min(Math.max(v, 0), m);
  const push = (x1, y1, x2, y2, score, cls) => {
    x1 = clamp(x1, sw); y1 = clamp(y1, sh); x2 = clamp(x2, sw); y2 = clamp(y2, sh);
    if (x2 - x1 > 1 && y2 - y1 > 1) list.push({ x1, y1, x2, y2, score, cls });
  };
  if (d1 > d2) {                                              // end2end: [1, N, 6] = x1,y1,x2,y2,score,cls
    for (let i = 0; i < d1; i++) {
      const o = i * d2, score = data[o + 4];
      if (score < minScore) continue;
      push((data[o] - meta.px) / meta.r, (data[o + 1] - meta.py) / meta.r, (data[o + 2] - meta.px) / meta.r, (data[o + 3] - meta.py) / meta.r, score, Math.round(data[o + 5]));
    }
    return list;
  }
  const nc = d1 - 4, N = d2;                                  // standar: [1, 4+nc, N], kotak = cx,cy,w,h
  for (let i = 0; i < N; i++) {
    let best = 0, bs = 0;
    for (let c = 0; c < nc; c++) { const v = data[(4 + c) * N + i]; if (v > bs) { bs = v; best = c; } }
    if (bs < minScore) continue;
    const cx = data[i], cy = data[N + i], w = data[2 * N + i], h = data[3 * N + i];
    push((cx - w / 2 - meta.px) / meta.r, (cy - h / 2 - meta.py) / meta.r, (cx + w / 2 - meta.px) / meta.r, (cy + h / 2 - meta.py) / meta.r, bs, best);
  }
  return nms(list);
}

async function runModel(src, sw, sh) {
  const meta = preprocess(src, sw, sh);
  const t0 = performance.now();
  const res = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', inputBuf, [1, 3, SIZE, SIZE]) });
  const ms = performance.now() - t0;
  return { out: res[session.outputNames[0]], meta, ms };
}

/* ----- 3c. Menggambar hasil & metrik ----- */
const kindOf = (d) => (String(CLASS_NAMES[d.cls] || '').toUpperCase().includes('UNRIPE') ? 'UNRIPE' : 'RIPE');

function fitOverlay(el, nw, nh) {                              // samakan kanvas dengan area gambar yang tampak (object-fit: contain)
  const cw = el.clientWidth, ch = el.clientHeight;
  if (!cw || !ch || !nw || !nh) return;
  const k = Math.min(cw / nw, ch / nh), w = nw * k, h = nh * k;
  Object.assign(overlay.style, { width: w + 'px', height: h + 'px', left: (cw - w) / 2 + 'px', top: (ch - h) / 2 + 'px' });
}
function clearOverlay() { overlay.getContext('2d').clearRect(0, 0, overlay.width, overlay.height); }

function drawDetections(dets, w, h) {
  const ctx = overlay.getContext('2d');
  if (overlay.width !== w || overlay.height !== h) { overlay.width = w; overlay.height = h; }
  ctx.clearRect(0, 0, w, h);
  const lw = Math.max(3, w / 200), fs = Math.max(14, w / 38);
  ctx.lineWidth = lw; ctx.font = `600 ${fs}px system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif`; ctx.textBaseline = 'top';
  for (const d of dets) {
    const kind = kindOf(d), color = kind === 'RIPE' ? '#F2B705' : '#5E9E2E';
    ctx.strokeStyle = color; ctx.strokeRect(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);
    const label = `${kind} ${Math.round(d.score * 100)}%`, tw = ctx.measureText(label).width + 12, ly = Math.max(0, d.y1 - fs - 8);
    ctx.fillStyle = color; ctx.fillRect(d.x1, ly, tw, fs + 8);
    ctx.fillStyle = kind === 'RIPE' ? '#4A3200' : '#FFFFFF'; ctx.fillText(label, d.x1 + 6, ly + 4);
  }
}

function renderList(dets) {
  preds.replaceChildren();
  dets.slice(0, 10).forEach((d, i) => {
    const kind = kindOf(d), li = document.createElement('li'), chip = document.createElement('span'), info = document.createElement('span');
    chip.className = 'chip ' + (kind === 'RIPE' ? 'ripe' : 'unripe'); chip.textContent = kind;
    info.textContent = `Pisang ${i + 1} · keyakinan ${Math.round(d.score * 100)}%`;
    li.append(chip, info); preds.appendChild(li);
  });
}

/* ----- 3d. Model 3D pendamping (kelas dominan) ----- */
let shownKind = null, candKind = null, candCount = 0;
function dominant(dets) {
  if (!dets.length) return null;
  const t = { RIPE: [0, 0], UNRIPE: [0, 0] };
  dets.forEach((d) => { const k = kindOf(d); t[k][0]++; t[k][1] += d.score; });
  return t.RIPE[0] !== t.UNRIPE[0] ? (t.RIPE[0] > t.UNRIPE[0] ? 'RIPE' : 'UNRIPE') : (t.RIPE[1] >= t.UNRIPE[1] ? 'RIPE' : 'UNRIPE');
}
function applyKind(kind) {
  if (kind === shownKind) return;
  shownKind = kind;
  document.querySelectorAll('.model-card').forEach((c) => {
    const v = c.querySelector('.viewer');
    c.classList.toggle('active', !!kind && v && v.dataset.model === MODEL_3D[kind]);
  });
  pendingModel = kind ? MODEL_3D[kind] : null;
  if (companion) {
    companion.setModel(pendingModel);
    if (!kind) companion.setMessage('Menunggu hasil deteksi…');
  }
  companionLabel.textContent = kind ? `Kelas dominan: ${kind === 'RIPE' ? 'RIPE (matang)' : 'UNRIPE (belum matang)'}` : 'Model 3D menyesuaikan kelas dominan';
}
function updateKind(dets, live) {
  const k = dominant(dets);
  if (!live) { applyKind(k); return; }
  if (k === shownKind) { candKind = k; candCount = 0; return; }   // stabilkan agar tidak berkedip antar frame
  if (k === candKind) candCount++; else { candKind = k; candCount = 1; }
  if (candCount >= (k ? 3 : 10)) applyKind(k);
}

/* ----- 3e. Uji gambar statis ----- */
function resetAll() {
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = currentFile = lastStatic = null;
  img.removeAttribute('src');
  if (!camActive) { stage.hidden = true; dropEmpty.hidden = false; }
  clearOverlay();
  preds.replaceChildren();
  fileInput.value = '';
  detectBtn.disabled = true; clearBtn.hidden = true; loading.hidden = true;
  mDet.textContent = '0'; mInf.textContent = '– ms'; mFps.textContent = '–';
  applyKind(null);
  if (!camActive) setMsg('Belum ada gambar. Unggah foto untuk memulai.');
}

fileInput.addEventListener('change', () => {
  const f = fileInput.files[0];
  if (!f) return;
  if (!ALLOWED.includes(f.type)) { resetAll(); setMsg('Format tidak didukung. Gunakan JPG, JPEG, atau PNG.', true); return; }
  if (f.size > MAX_BYTES) { resetAll(); setMsg('Ukuran file melebihi 5 MB.', true); return; }
  if (camActive) stopCamera();
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  currentFile = f; lastStatic = null;
  objectUrl = URL.createObjectURL(f);       // pratinjau lokal, foto tidak dikirim ke mana pun
  preds.replaceChildren();
  img.onload = () => {
    stage.hidden = false; dropEmpty.hidden = true; img.hidden = false; video.hidden = true;
    clearOverlay(); fitOverlay(img, img.naturalWidth, img.naturalHeight);
    detectBtn.disabled = false; clearBtn.hidden = false;
    setMsg('Gambar siap. Tekan “Mulai Deteksi”.');
  };
  img.onerror = () => { resetAll(); setMsg('File tidak dapat dibaca sebagai gambar.', true); };
  img.src = objectUrl;
});
clearBtn.addEventListener('click', resetAll);

function showStatic() {
  const { out, meta, ms } = lastStatic, w = img.naturalWidth, h = img.naturalHeight;
  const dets = decode(out, meta, threshold(), w, h);
  fitOverlay(img, w, h); drawDetections(dets, w, h); renderList(dets); updateKind(dets, false);
  mDet.textContent = dets.length; mInf.textContent = ms.toFixed(1) + ' ms'; mFps.textContent = '–';
  setMsg(dets.length ? `${dets.length} pisang terdeteksi.` : 'Tidak ada pisang yang terdeteksi pada gambar ini.');
}
function redrawStatic() { if (lastStatic && !camActive) showStatic(); }

detectBtn.addEventListener('click', async () => {
  if (!currentFile) return;
  detectBtn.disabled = true; loading.hidden = false; preds.replaceChildren(); setMsg('');
  try {
    if (!(await ensureModel())) { setMsg('Model belum siap. Periksa status di Model Setup.', true); return; }
    const r = await runModel(img, img.naturalWidth, img.naturalHeight);
    lastStatic = r; showStatic();
  } catch (err) {
    console.error(err);
    setMsg('Deteksi gagal: ' + err.message + '. Coba lagi beberapa saat.', true);
  } finally { loading.hidden = true; detectBtn.disabled = false; }
});

conf.addEventListener('input', () => { confVal.textContent = threshold().toFixed(2); redrawStatic(); });
window.addEventListener('resize', () => {
  if (camActive && video.videoWidth) fitOverlay(video, video.videoWidth, video.videoHeight);
  else if (img.naturalWidth && !stage.hidden) fitOverlay(img, img.naturalWidth, img.naturalHeight);
});

/* ----- 3f. Kamera real-time ----- */
async function startCamera() {
  if (camActive) return;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) { setMsg('Browser ini tidak mendukung kamera (butuh HTTPS atau localhost).', true); return; }
  startCamBtn.disabled = true;
  try {
    if (!(await ensureModel())) { setMsg('Model belum siap. Periksa status di Model Setup.', true); startCamBtn.disabled = false; return; }
    camStream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: { ideal: camSource.value }, width: { ideal: 1280 }, height: { ideal: 720 } } });
    video.srcObject = camStream;
    await video.play();
    camActive = true;
    stage.hidden = false; dropEmpty.hidden = true; img.hidden = true; video.hidden = false;
    detectBtn.disabled = true; clearOverlay(); preds.replaceChildren();
    stopCamBtn.disabled = false; setMsg('Kamera aktif. Arahkan ke pisang.');
    liveLoop();
  } catch (err) {
    console.error(err);
    startCamBtn.disabled = false;
    setMsg(err.name === 'NotAllowedError' ? 'Izin kamera ditolak. Aktifkan izin kamera di browser.' : 'Kamera gagal dibuka: ' + err.message, true);
  }
}
function stopCamera() {
  camActive = false;
  if (camStream) camStream.getTracks().forEach((t) => t.stop());
  camStream = null; video.srcObject = null; video.hidden = true;
  startCamBtn.disabled = false; stopCamBtn.disabled = true;
  clearOverlay(); preds.replaceChildren();
  mDet.textContent = '0'; mInf.textContent = '– ms'; mFps.textContent = '–';
  applyKind(null);
  if (currentFile && img.src) { img.hidden = false; stage.hidden = false; dropEmpty.hidden = true; detectBtn.disabled = false; fitOverlay(img, img.naturalWidth, img.naturalHeight); setMsg('Kamera dihentikan. Gambar siap dideteksi.'); }
  else { stage.hidden = true; dropEmpty.hidden = false; setMsg('Kamera dihentikan.'); }
}
async function liveLoop() {
  let fps = 0, last = performance.now(), lastUi = 0;
  while (camActive) {
    await new Promise(requestAnimationFrame);
    if (!camActive) break;
    if (!session || video.readyState < 2 || !video.videoWidth) continue;
    try {
      const w = video.videoWidth, h = video.videoHeight;
      const { out, meta, ms } = await runModel(video, w, h);
      if (!camActive) break;
      const dets = decode(out, meta, threshold(), w, h);
      fitOverlay(video, w, h); drawDetections(dets, w, h);
      const now = performance.now(), inst = 1000 / Math.max(1, now - last); last = now;
      fps = fps ? fps * 0.85 + inst * 0.15 : inst;                 // FPS dihaluskan
      mDet.textContent = dets.length; mInf.textContent = ms.toFixed(1) + ' ms'; mFps.textContent = fps.toFixed(1);
      updateKind(dets, true);
      if (now - lastUi > 300) { lastUi = now; renderList(dets); }
    } catch (err) { console.error(err); setMsg('Inferensi kamera gagal: ' + err.message, true); stopCamera(); break; }
  }
}
startCamBtn.addEventListener('click', startCamera);
stopCamBtn.addEventListener('click', stopCamera);
camSource.addEventListener('change', async () => { if (camActive) { stopCamera(); await startCamera(); } });
drop.addEventListener('click', (e) => { if (camActive) e.preventDefault(); });   // klik pratinjau kamera jangan membuka dialog file
document.addEventListener('visibilitychange', () => { if (document.hidden && camActive) stopCamera(); });
