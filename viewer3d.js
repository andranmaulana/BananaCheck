/* viewer3d.js — model 3D (Three.js). Dimuat dari script.js lewat import() agar kegagalan di sini tidak merusak navbar & area deteksi. */
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
let sharedEnv = null; // lingkungan cahaya dibuat sekali, dipakai dua viewer

function createViewer(box) {
  const status = box.querySelector('.v-status');
  const showError = (text) => { status.textContent = text; status.classList.add('error'); status.hidden = false; };

  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'low-power' });
  } catch (err) {
    showError('Browser ini tidak mendukung WebGL, model 3D tidak dapat ditampilkan.');
    return null;
  }
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  box.prepend(renderer.domElement);

  const scene = new THREE.Scene();
  if (!sharedEnv) {
    const pmrem = new THREE.PMREMGenerator(renderer);
    sharedEnv = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  }
  scene.environment = sharedEnv;

  // Cahaya utama + bayangan lembut
  const sun = new THREE.DirectionalLight(0xffffff, 2.2);
  sun.position.set(3, 5, 3);
  sun.castShadow = true;
  sun.shadow.mapSize.set(1024, 1024);
  sun.shadow.radius = 6;
  Object.assign(sun.shadow.camera, { left: -3, right: 3, top: 3, bottom: -3, near: 0.5, far: 15 });
  scene.add(sun, new THREE.HemisphereLight(0xfff4d6, 0xd9c28a, 0.6));

  const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 100);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enablePan = false;
  controls.enableZoom = false;          // agar scroll halaman di HP tidak terganggu
  controls.enableDamping = true;
  controls.autoRotate = !reduceMotion;
  controls.autoRotateSpeed = 1.6;       // rotasi halus
  renderer.domElement.style.touchAction = 'pan-y'; // geser horizontal = putar, vertikal = scroll
  controls.minPolarAngle = 0.5;
  controls.maxPolarAngle = Math.PI / 2 + 0.1;

  let radius = 1; // radius bola pembungkus model, dipakai untuk menyesuaikan kamera
  function fitCamera() {
    const w = box.clientWidth, h = box.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    const vFov = THREE.MathUtils.degToRad(camera.fov);
    const hFov = 2 * Math.atan(Math.tan(vFov / 2) * camera.aspect);
    const dist = (radius / Math.sin(Math.min(vFov, hFov) / 2)) * 1.08; // margin agar tidak terpotong
    const dir = camera.position.clone().sub(controls.target);
    if (dir.lengthSq() < 1e-6) dir.set(0, 0.35, 1);
    camera.position.copy(controls.target).add(dir.normalize().multiplyScalar(dist));
    camera.near = dist / 50; camera.far = dist * 10;
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(fitCamera).observe(box);

  // Muat GLB (bisa dipanggil ulang untuk mengganti model)
  const cache = new Map();   // url -> {pivot, ground, radius}
  let current = null, token = 0;
  function mount(entry) {
    if (current) scene.remove(current.pivot, current.ground);
    current = entry;
    scene.add(entry.pivot, entry.ground);
    radius = entry.radius;
    camera.position.set(0, 0.35, 1);
    status.hidden = true;
    fitCamera();
  }
  function loadModel(url) {
    const my = ++token;
    if (!url) {
      if (current) scene.remove(current.pivot, current.ground);
      current = null;
      return;
    }
    if (cache.has(url)) { mount(cache.get(url)); return; }
    status.classList.remove('error'); status.textContent = 'Memuat model…'; status.hidden = false;
    new GLTFLoader().load(
      url,
      (gltf) => {
       try {
        const model = gltf.scene;
        model.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = false; } });

        // Normalisasi ukuran & posisi: pusatkan, skala ke tinggi/lebar maks = 2
        const bb = new THREE.Box3().setFromObject(model);
        const size = bb.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z) || 1;
        model.scale.setScalar(2 / maxDim);
        bb.setFromObject(model);
        const center = bb.getCenter(new THREE.Vector3());
        model.position.sub(center);                 // pusat di (0,0,0)
        const pivot = new THREE.Group();
        pivot.add(model);

        const finalSize = new THREE.Box3().setFromObject(pivot).getSize(new THREE.Vector3());

        // Bidang bayangan di bawah model
        const ground = new THREE.Mesh(new THREE.PlaneGeometry(10, 10), new THREE.ShadowMaterial({ opacity: 0.22 }));
        ground.rotation.x = -Math.PI / 2;
        ground.position.y = -finalSize.y / 2 - 0.05;
        ground.receiveShadow = true;

        const entry = { pivot, ground, radius: finalSize.length() / 2 };
        cache.set(url, entry);
        if (my === token) mount(entry);   // abaikan jika sudah diganti model lain
       } catch (err) { console.error(err); showError('Model terunduh tetapi gagal ditampilkan: ' + err.message); }
      },
      (e) => {
        if (my !== token) return;
        status.textContent = e.lengthComputable && e.total
          ? `Memuat model… ${Math.round((e.loaded / e.total) * 100)}%`
          : `Memuat model… ${Math.round(e.loaded / 1024)} KB`;
      },
      (err) => { console.error(err); showError(`Model ${box.dataset.name || url} gagal dimuat. Pastikan file ${url} ada di folder yang sama dan website dibuka lewat server (bukan file://).`); }
    );
  }
  if (box.dataset.model) loadModel(box.dataset.model);

  // Render hanya saat terlihat di layar (hemat baterai HP)
  let visible = true;
  new IntersectionObserver(([en]) => { visible = en.isIntersecting; }).observe(box);
  renderer.setAnimationLoop(() => {
    if (!visible || document.hidden) return;
    controls.update();
    renderer.render(scene, camera);
  });
  return {
    setModel: loadModel,
    setMessage(text) { status.classList.remove('error'); status.textContent = text; status.hidden = false; },
  };
}
export function initViewers() {
  // Viewer hero: yang punya data-model. Viewer pendamping (#companion) dibuat terpisah.
  document.querySelectorAll('.viewer[data-model]').forEach((b) => createViewer(b));
}
// Viewer pendamping di kartu hasil: modelnya diganti dari script.js sesuai kelas dominan.
export function initCompanion(box) { return createViewer(box); }
