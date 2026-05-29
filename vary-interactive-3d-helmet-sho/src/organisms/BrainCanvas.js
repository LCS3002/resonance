import * as THREE from 'three';
import { RGBELoader } from 'three/addons/loaders/RGBELoader.js';
import { ENDPOINTS } from '../config/api.js';

const HDRI_LIGHT = 'https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/citrus_orchard_road_puresky_1k.hdr';
const HDRI_DARK  = 'https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/qwantani_dusk_2_1k.hdr';

export function initBrainCanvas(canvasEl, modeToggleEl) {
  const renderer = new THREE.WebGLRenderer({ canvas: canvasEl, antialias: true, alpha: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.15;
  renderer.setClearColor(0x000000, 0);

  const scene  = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, window.innerWidth / window.innerHeight, 0.01, 100);
  camera.position.set(0, 0, 3.2);

  scene.add(new THREE.AmbientLight(0xfff6ee, 0.55));
  const dl1 = new THREE.DirectionalLight(0xfff5e8, 2.8); dl1.position.set(4, 5, 4);   scene.add(dl1);
  const dl2 = new THREE.DirectionalLight(0x9ab0cc, 1.0); dl2.position.set(-4, 1, 1);  scene.add(dl2);
  const dl3 = new THREE.DirectionalLight(0xff8855, 1.2); dl3.position.set(1, -3, -5); scene.add(dl3);
  const dl4 = new THREE.DirectionalLight(0xffffff, 0.4); dl4.position.set(0, 6, 1);   scene.add(dl4);

  const hdriCache = {};
  function loadHDRI(url) {
    return new Promise(resolve => {
      if (hdriCache[url]) { resolve(hdriCache[url]); return; }
      new RGBELoader().load(url, tex => {
        tex.mapping = THREE.EquirectangularReflectionMapping;
        hdriCache[url] = tex;
        resolve(tex);
      });
    });
  }
  loadHDRI(HDRI_LIGHT).then(t => { scene.environment = t; });
  loadHDRI(HDRI_DARK);

  let isDark = false;
  modeToggleEl.addEventListener('click', async () => {
    isDark = !isDark;
    document.body.classList.toggle('dark-mode', isDark);
    scene.environment = await loadHDRI(isDark ? HDRI_DARK : HDRI_LIGHT);
    renderer.setClearColor(0x000000, 0);
    renderer.toneMappingExposure = isDark ? 1.3 : 1.0;
    updateBrainColors();
  });

  let brain = null, sulcData = null, regionWeights = null;
  let nVerts = 0, colorAttr = null;

  function hotColor(t) {
    t = Math.max(0, Math.min(1, t));
    if (t < 0.33) return [t * 3, 0, 0];
    if (t < 0.66) return [1, (t - 0.33) * 3, 0];
    return [1, 1, (t - 0.66) * 3];
  }

  function vertexColor(sulcVal, activation) {
    const lo = isDark ? 0.06 : 0.20;
    const hi = isDark ? 0.78 : 0.97;
    const base = lo + sulcVal * (hi - lo);
    const r = Math.min(1, base + sulcVal * (isDark ? 0.06 : 0.10));
    const g = base + sulcVal * 0.01;
    const b = Math.max(0, base - sulcVal * (isDark ? 0.08 : 0.16));
    if (activation < 0.015) return [r, g, b];
    const [hr, hg, hb] = hotColor(activation);
    const blend = Math.min(1, activation * 2.2);
    return [r + (hr - r) * blend, g + (hg - g) * blend, b + (hb - b) * blend];
  }

  function sc(x) { return Math.max(0, Math.min(1, x)); }

  function computeRegionWeights(posArr, nLeft) {
    const n  = posArr.length / 3;
    const rw = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const x = posArr[i*3], y = posArr[i*3+1], z = posArr[i*3+2];
      const visual = sc((-y - 0.3) / 0.35) * sc(1 - Math.abs(x) * 1.2) * sc(1 - Math.abs(z) * 1.5);
      const isLeft = i < nLeft;
      const lang   = isLeft
        ? sc((Math.abs(x) - 0.15) / 0.25) * sc(1 - (-y - 0.1) / 0.3) * sc((z + 0.3) / 0.4) * sc(1 - (z - 0.35) / 0.2)
        : 0;
      const prefrontal = sc((y - 0.55) / 0.25) * sc((z - 0.1) / 0.35);
      rw[i*3] = visual; rw[i*3+1] = lang; rw[i*3+2] = prefrontal;
    }
    return rw;
  }

  let activationVisual = 0, activationLanguage = 0, activationPrefrontal = 0;
  let targetVisual     = 0, targetLanguage     = 0, targetPrefrontal     = 0;
  let useApiActivation = false;

  function updateBrainColors() {
    if (!colorAttr) return;
    const col = colorAttr.array;
    for (let i = 0; i < nVerts; i++) {
      const sulc = sulcData[i];
      const act  = Math.min(1,
        regionWeights[i*3  ] * activationVisual     * 0.85 +
        regionWeights[i*3+1] * activationLanguage   * 0.85 +
        regionWeights[i*3+2] * activationPrefrontal * 0.85
      );
      const [r, g, b] = vertexColor(sulc, act);
      col[i*3] = r; col[i*3+1] = g; col[i*3+2] = b;
    }
    colorAttr.needsUpdate = true;
  }

  window.setApiActivation = function(lang, vis, pre) {
    useApiActivation = true;
    activationLanguage = activationVisual = activationPrefrontal = 0.95;
    updateBrainColors();
    canvasEl.classList.add('pulsing');
    canvasEl.addEventListener('animationend', () => canvasEl.classList.remove('pulsing'), { once: true });
    setTimeout(() => { targetLanguage = lang; targetVisual = vis; targetPrefrontal = pre; }, 350);
  };

  fetch(ENDPOINTS.brainMesh).then(r => r.json()).then(data => {
    const posArr = new Float32Array(data.vertices);
    sulcData      = new Float32Array(data.sulc);
    nVerts        = posArr.length / 3;
    regionWeights = computeRegionWeights(posArr, data.n_left);

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
    geo.setIndex(new THREE.BufferAttribute(new Uint32Array(data.faces), 1));
    geo.computeVertexNormals();

    const initColors = new Float32Array(nVerts * 3);
    colorAttr = new THREE.BufferAttribute(initColors, 3);
    geo.setAttribute('color', colorAttr);
    updateBrainColors();

    brain = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
      vertexColors: true, roughness: 0.58, metalness: 0.04,
      side: THREE.FrontSide, flatShading: false,
    }));
    scene.add(brain);

    const ptIdx = [];
    for (let i = 0; i < nVerts; i += 8) ptIdx.push(i);
    const ptPos = new Float32Array(ptIdx.length * 3);
    const ptCol = new Float32Array(ptIdx.length * 3);
    for (let j = 0; j < ptIdx.length; j++) {
      const i = ptIdx[j];
      ptPos[j*3] = posArr[i*3]; ptPos[j*3+1] = posArr[i*3+1]; ptPos[j*3+2] = posArr[i*3+2];
      ptCol[j*3] = ptCol[j*3+1] = ptCol[j*3+2] = 0.75;
    }
    const ptGeo = new THREE.BufferGeometry();
    ptGeo.setAttribute('position', new THREE.BufferAttribute(ptPos, 3));
    const ptColAttr = new THREE.BufferAttribute(ptCol, 3);
    ptGeo.setAttribute('color', ptColAttr);
    const brainPts = new THREE.Points(ptGeo, new THREE.PointsMaterial({
      vertexColors: true, size: 0.006, sizeAttenuation: true, transparent: true, opacity: 0.55,
    }));
    scene.add(brainPts);
    brain._pts = brainPts;
    brain._ptColAttr = ptColAttr;
    brain._ptIdx = ptIdx;

    document.getElementById('loadingOverlay').classList.add('hidden');
    setTimeout(() => { document.getElementById('loadingOverlay').style.display = 'none'; }, 700);
    setTimeout(() => { document.querySelector('.hero .fade-in')?.classList.add('visible'); }, 300);
  });

  function smoothstep(a, b, t) {
    const x = Math.max(0, Math.min(1, (t - a) / (b - a)));
    return x * x * (3 - 2 * x);
  }
  function lerp(a, b, t) { return a + (b - a) * t; }

  const sections = [
    { scrollStart: 0,    scrollEnd: 0.15, posX:  0.2, posY: 0.0, rotX:  0.05, rotY: 0.10,      camZ: 3.3, scale: 1.0  },
    { scrollStart: 0.15, scrollEnd: 0.35, posX:  0.9, posY: 0,   rotX: -0.05, rotY: -0.5,       camZ: 2.8, scale: 1.05 },
    { scrollStart: 0.35, scrollEnd: 0.55, posX: -0.8, posY: 0.1, rotX:  0.3,  rotY: Math.PI,    camZ: 2.6, scale: 1.05 },
    { scrollStart: 0.55, scrollEnd: 0.65, posX:  0,   posY: 0,   rotX: -0.4,  rotY: 0.2,        camZ: 3.0, scale: 0.95 },
    { scrollStart: 0.65, scrollEnd: 0.80, posX:  0.7, posY: 0,   rotX: -0.3,  rotY: 0.6,        camZ: 2.5, scale: 1.08 },
    { scrollStart: 0.80, scrollEnd: 1.0,  posX: -0.5, posY: 0,   rotX:  0.1,  rotY: Math.PI*2,  camZ: 3.2, scale: 0.95 },
  ];

  let scrollProgress = 0, targetScroll = 0;
  let isDragging = false, dragStartX = 0, dragStartY = 0;
  let orbitX = 0, orbitY = 0, targetOrbitX = 0, targetOrbitY = 0;
  const DRAG_SENS = 0.006, ORBIT_DECAY = 0.04;

  window.addEventListener('pointerdown', e => { isDragging = true; dragStartX = e.clientX; dragStartY = e.clientY; });
  window.addEventListener('pointermove', e => {
    if (!isDragging) return;
    targetOrbitX = (e.clientX - dragStartX) * DRAG_SENS;
    targetOrbitY = (e.clientY - dragStartY) * DRAG_SENS;
  });
  window.addEventListener('pointerup', () => { isDragging = false; dragStartX = dragStartY = 0; });
  window.addEventListener('scroll', () => {
    const sh = document.documentElement.scrollHeight - window.innerHeight;
    targetScroll = sh > 0 ? window.scrollY / sh : 0;
  });
  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  const fadeEls = document.querySelectorAll('.fade-in');
  const observer = new IntersectionObserver(
    entries => entries.forEach(e => { if (e.isIntersecting) e.target.classList.add('visible'); }),
    { threshold: 0.15 }
  );
  fadeEls.forEach(el => observer.observe(el));

  let colorTimer = 0;

  function animate() {
    requestAnimationFrame(animate);
    scrollProgress += (targetScroll - scrollProgress) * 0.06;

    if (useApiActivation) {
      activationLanguage   += (targetLanguage   - activationLanguage)   * 0.03;
      activationVisual     += (targetVisual     - activationVisual)     * 0.03;
      activationPrefrontal += (targetPrefrontal - activationPrefrontal) * 0.03;
    } else {
      targetLanguage   = smoothstep(0.12, 0.38, scrollProgress);
      targetVisual     = smoothstep(0.32, 0.58, scrollProgress);
      targetPrefrontal = smoothstep(0.58, 0.82, scrollProgress);
      activationLanguage   += (targetLanguage   - activationLanguage)   * 0.04;
      activationVisual     += (targetVisual     - activationVisual)     * 0.04;
      activationPrefrontal += (targetPrefrontal - activationPrefrontal) * 0.04;
    }

    colorTimer++;
    if (colorTimer % 2 === 0) updateBrainColors();

    if (brain) {
      let cur = sections[0], nxt = sections[1], localT = 0;
      for (let i = 0; i < sections.length; i++) {
        if (scrollProgress >= sections[i].scrollStart && scrollProgress <= sections[i].scrollEnd) {
          cur = sections[i];
          nxt = sections[Math.min(i + 1, sections.length - 1)];
          localT = smoothstep(cur.scrollStart, cur.scrollEnd, scrollProgress);
          break;
        }
        if (i === sections.length - 1) { cur = nxt = sections[i]; localT = 1; }
      }

      if (isDragging) { orbitX += (targetOrbitX - orbitX) * 0.12; orbitY += (targetOrbitY - orbitY) * 0.12; }
      else { orbitX += -orbitX * ORBIT_DECAY; orbitY += -orbitY * ORBIT_DECAY; targetOrbitX += -targetOrbitX * ORBIT_DECAY; targetOrbitY += -targetOrbitY * ORBIT_DECAY; }

      brain.position.x = lerp(cur.posX, nxt.posX, localT);
      brain.position.y = lerp(cur.posY, nxt.posY, localT);
      brain.rotation.x = lerp(cur.rotX, nxt.rotX, localT) + orbitY;
      brain.rotation.y = lerp(cur.rotY, nxt.rotY, localT) + Math.sin(Date.now() * 0.00025) * 0.04 + orbitX;
      brain.scale.setScalar(lerp(cur.scale, nxt.scale, localT));
      camera.position.z = lerp(cur.camZ, nxt.camZ, localT);

      if (brain._pts) {
        brain._pts.position.copy(brain.position);
        brain._pts.rotation.copy(brain.rotation);
        brain._pts.scale.copy(brain.scale);
        if (colorTimer % 4 === 0 && brain._ptIdx) {
          const mc = colorAttr.array;
          const pc = brain._ptColAttr.array;
          const idx = brain._ptIdx;
          for (let j = 0; j < idx.length; j++) {
            const i = idx[j];
            pc[j*3]   = Math.min(1, mc[i*3]   * 1.12);
            pc[j*3+1] = Math.min(1, mc[i*3+1] * 1.10);
            pc[j*3+2] = Math.min(1, mc[i*3+2] * 1.08);
          }
          brain._ptColAttr.needsUpdate = true;
        }
      }
    }

    renderer.render(scene, camera);
  }
  animate();
}
