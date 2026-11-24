# 🍌 BananaCheck Deteksi Kematangan Pisang Berbasis YOLO26 & Web 3D

**BananaCheck** adalah aplikasi web berbasis *Computer Vision* yang mendeteksi tingkat kematangan buah pisang secara *real-time* langsung di browser (*client-side*, tanpa server backend), dilengkapi visualisasi model 3D interaktif.

## 🔗 Live Demo
👉 **[https://andranmaulana.github.io/BananaCheck/](https://andranmaulana.github.io/BananaCheck/)**

---

## ✨ Fitur Utama
- **Deteksi real-time di browser:** inferensi YOLO26 memakai **ONNX Runtime Web (WebGPU, otomatis turun ke WASM)** dari kamera HP/laptop maupun gambar statis. Foto dan video tidak dikirim ke server mana pun.
- **Dua kelas kematangan:**
  - `ripe` — pisang matang (kuning)
  - `unripe` — pisang mentah (hijau)
- **Visualisasi 3D interaktif:** model `banana_ripe.glb` / `banana_unripe.glb` tampil sesuai kelas yang paling dominan terdeteksi.
- **Pengaturan fleksibel:** *Confidence Threshold* (default `0.25`), kamera depan/belakang, muat model `.onnx` lokal, serta metrik *Detections*, *Inference (ms)*, dan *FPS*.

---

## 🚀 Cara Menggunakan
1. Buka website. Model default (`models/best.onnx`) dimuat otomatis; status berubah menjadi **Model ready · WebGPU/WASM**.
2. **Kamera:** pilih Back/Front camera → **Start camera** → izinkan akses kamera.
3. **Gambar:** **Pilih Gambar** (JPG/PNG, maks 5 MB) → **Mulai Deteksi**.
4. Atur slider **Confidence threshold** untuk menyaring hasil.
5. *(Opsional)* Pakai model sendiri: **Choose .onnx** → **Load model**. Tombol **Use default** mengembalikan ke model bawaan.

## 💻 Menjalankan Secara Lokal
Website harus dibuka lewat server (bukan `file://`):
```bash
cd BananaCheck
python -m http.server 8000
# buka http://localhost:8000
```
Kamera hanya berfungsi di **HTTPS** atau `localhost`. Deploy via GitHub Pages sudah memenuhi syarat ini.

---

## 🧠 Detail Model
| Item | Nilai |
|---|---|
| Arsitektur | Ultralytics YOLO26n (detect) |
| Kelas | `0: ripe`, `1: unripe` |
| Input | `1 × 3 × 640 × 640` (letterbox, RGB, 0–1) |
| Output | `1 × 6 × 8400` (cx, cy, w, h, skor ripe, skor unripe) |
| Pasca-proses | NMS per kelas di JavaScript (IoU 0,45) |
| Format | ONNX (`models/best.onnx`) |

## 📊 Informasi Dataset & Pelatihan
- **Anotasi & augmentasi:** Roboflow
- **Total dataset:** 2.410 gambar (setelah augmentasi 3×)
  - Training: 2.109 · Validation: 201 · Test: 100
- **Augmentasi:** Flip horizontal · Rotasi 90° CW & CCW · Brightness ±15% · Exposure ±10% · Blur hingga 2,5 px · Noise hingga 0,1% piksel
- **Resolusi input (`IMG_SIZE`):** 640 × 640 piksel

---

## 📁 Struktur Direktori
```text
BananaCheck/
├── index.html          # Struktur antarmuka utama
├── style.css           # Styling responsif
├── script.js           # Inferensi ONNX Runtime Web, kontrol kamera, pasca-proses
├── viewer3d.js         # Renderer objek 3D (.glb) dengan Three.js
├── banana_ripe.glb     # Model 3D pisang matang
├── banana_unripe.glb   # Model 3D pisang mentah
├── models/
│   └── best.onnx       # Bobot YOLO26 hasil training (ONNX)
└── README.md           # Dokumentasi proyek
```

## 🛠️ Teknologi
HTML · CSS · JavaScript · [ONNX Runtime Web](https://onnxruntime.ai/) · [Three.js](https://threejs.org/) · [Ultralytics YOLO26](https://docs.ultralytics.com/) · Roboflow · GitHub Pages

## ⚠️ Catatan
- Warna hanyalah petunjuk visual dan bukan jaminan sempurna tingkat kematangan.
- Library ONNX Runtime Web dan Three.js dimuat dari CDN jsDelivr, jadi perlu koneksi internet saat pertama kali membuka halaman.
- Model dilatih dengan Ultralytics (lisensi AGPL-3.0); perhatikan ketentuan lisensinya bila proyek dipublikasikan atau dikomersialkan.

## 👥 Tim Kelompok Gerengse
- Andran Maulana
- Hoerul Anam
- Messi Bima Putra
- Rantam Muhammad Farhan
- Muhammad Faisal Hasan

*Educational AI Project © 2026 BananaCheck*
