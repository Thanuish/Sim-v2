// MuJoCo VR teleop client: renders the server's MuJoCo scene in WebXR and
// streams Quest controller poses back. Physics, IK and recording all run in Python.
import * as THREE from 'three';
import { VRButton } from 'three/addons/webxr/VRButton.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { XRControllerModelFactory } from 'three/addons/webxr/XRControllerModelFactory.js';

const PROTOCOL_VERSION = 5;   // must match server.py
// ?lite in the URL turns off shadows and reflections (for older headsets)
const LITE = new URLSearchParams(location.search).has('lite');
let worldEpoch = 0;           // bumped whenever the VR world is re-aligned (recenter)

// ---------------------------------------------------------------- renderer
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;   // photographic highlight roll-off
renderer.toneMappingExposure = 1.0;
renderer.shadowMap.enabled = !LITE;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.xr.enabled = true;
renderer.xr.setReferenceSpaceType('local-floor');
renderer.xr.setFoveation(1);
document.body.appendChild(renderer.domElement);
document.getElementById('xrrow').appendChild(VRButton.createButton(renderer));
const vrBtn = document.querySelector('#xrrow button');
if (vrBtn) { vrBtn.style.position = 'static'; vrBtn.style.transform = 'none'; vrBtn.style.left = 'auto'; }

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xd9dadb);

// Image-based lighting from a simple procedural room (bright ceiling panel, light walls,
// warm floor): gives metal and plastic parts believable reflections and soft ambient light.
function makeEnvironment() {
  const env = new THREE.Scene();
  const box = new THREE.Mesh(new THREE.BoxGeometry(6, 3, 6),
    new THREE.MeshBasicMaterial({ color: 0xbfbcb6, side: THREE.BackSide }));
  box.position.y = 1.5;
  env.add(box);
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(6, 6), new THREE.MeshBasicMaterial({ color: 0x7a5a3d }));
  floor.rotation.x = -Math.PI / 2; floor.position.y = 0.01;
  env.add(floor);
  const panel = (w, h, x, y, z, rx, c) => {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ color: c, side: THREE.DoubleSide }));
    m.position.set(x, y, z); m.rotation.x = rx; env.add(m);
  };
  panel(1.2, 0.6, 0, 2.95, -0.45, Math.PI / 2, new THREE.Color(6, 6, 5.8));    // ceiling fixture
  panel(3.0, 1.4, 0, 1.6, -2.95, 0, new THREE.Color(1.6, 1.7, 1.9));           // window-like wall glow
  const pmrem = new THREE.PMREMGenerator(renderer);
  const tex = pmrem.fromScene(env, 0.035).texture;
  pmrem.dispose();
  return tex;
}
if (!LITE) { scene.environment = makeEnvironment(); scene.environmentIntensity = 0.55; }
scene.add(new THREE.HemisphereLight(0xfaf8f2, 0x6b5a48, LITE ? 1.4 : 0.55));
// Ceiling fixture above the table (matches the MuJoCo 'ceiling' light), casting soft shadows
const sun = new THREE.DirectionalLight(0xfff8ee, 2.2);
sun.castShadow = !LITE;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.left = -1.1; sun.shadow.camera.right = 1.1;
sun.shadow.camera.top = 1.1; sun.shadow.camera.bottom = -1.1;
sun.shadow.camera.near = 0.5; sun.shadow.camera.far = 4;
sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.01;
sun.shadow.radius = 4;
scene.add(sun);
scene.add(sun.target);
const fill = new THREE.DirectionalLight(0xe8eef8, 0.45);
fill.position.set(-1.5, 2.0, 1.5);
scene.add(fill);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.05, 30);
camera.position.set(0, 1.6, 0.12);    // just behind the robot's "eyes"
const orbit = new OrbitControls(camera, renderer.domElement);
orbit.target.set(0, 0.8, -0.75);
orbit.update();

// ------------------------------------------------------- MuJoCo frame setup
// MuJoCo: Z-up, +x forward, +y left.  Three/WebXR: Y-up, -Z forward, +X right.
// Columns = where the MuJoCo x, y, z axes end up in three.js.
const world = new THREE.Group();          // moved by recenter / stick sliding
scene.add(world);
const mjRoot = new THREE.Group();
mjRoot.matrixAutoUpdate = false;
mjRoot.matrix.makeBasis(new THREE.Vector3(0, 0, -1), new THREE.Vector3(-1, 0, 0), new THREE.Vector3(0, 1, 0));
world.add(mjRoot);
// First-person: your eyes are placed at the robot's "head", between the two arm bases.
// Keep in sync with HEAD_POS in scene.py.
const OPERATOR_EYE_MJ = new THREE.Vector3(-0.30, 0, 1.52);
// Personal viewpoint adjustment (MuJoCo frame, metres), set live in VR with trigger + stick
// and remembered in this browser.
const viewOffset = new THREE.Vector3();
try { const v = JSON.parse(localStorage.getItem('viewOffset') || 'null'); if (v) viewOffset.set(v[0], v[1], v[2]); } catch (_) {}
function saveViewOffset() {
  try { localStorage.setItem('viewOffset', JSON.stringify(viewOffset.toArray())); } catch (_) {}
}
const eyeMJ = () => OPERATOR_EYE_MJ.clone().add(viewOffset);
function defaultPlacement() {      // eye horizontally at the origin, true heights
  world.rotation.set(0, 0, 0);
  const e = new THREE.Vector3(OPERATOR_EYE_MJ.x, OPERATOR_EYE_MJ.y, 0).applyMatrix4(mjRoot.matrix);
  world.position.copy(e).multiplyScalar(-1);
}
defaultPlacement();
function placeLights() {       // keep the shadow-casting light above the table, whatever the recenter
  world.updateMatrixWorld(true);
  const tbl = new THREE.Vector3(0.45, 0.0, 0.75).applyMatrix4(mjRoot.matrixWorld);
  const lamp = new THREE.Vector3(0.25, 0.35, 2.7).applyMatrix4(mjRoot.matrixWorld);
  sun.position.copy(lamp);
  sun.target.position.copy(tbl);
  sun.target.updateMatrixWorld();
}
placeLights();

// ------------------------------------------------------------ scene loading
let bodyObjs = [];
let cloths = [];                 // deformable garments: {geo, W, coarse, fine, nvert}
let sceneTextures = [];
const texLoader = new THREE.TextureLoader();
const matCache = new Map();

// PBR hints for the Menagerie robot materials (MuJoCo only gives colours)
function pbrHints(name, m) {
  const n = (name || '').toLowerCase();
  if (/metal|alumin/.test(n)) return { roughness: 0.32, metalness: 0.85 };
  if (/linkgray/.test(n)) return { roughness: 0.38, metalness: 0.35 };        // anodised tube
  if (/urblue/.test(n)) return { roughness: 0.42, metalness: 0.0 };           // painted caps
  if (/silicone|black|jointgray/.test(n)) return { roughness: 0.62, metalness: 0.0 };
  if (/denim/.test(n)) return { roughness: 0.95, metalness: 0.0 };
  if (/wood/.test(n)) return { roughness: 0.55, metalness: 0.0 };
  if (/laminate/.test(n)) return { roughness: 0.5, metalness: 0.0 };
  if (/wall|skirt/.test(n)) return { roughness: 0.9, metalness: 0.0 };
  return { roughness: m ? THREE.MathUtils.clamp(m.roughness, 0.05, 1) : 0.55,
           metalness: m ? THREE.MathUtils.clamp(m.metallic, 0, 1) : 0.1 };
}
function textureFor(i, repeatX = 1, repeatY = 1) {
  if (i < 0 || !sceneTextures[i]) return null;
  const t = sceneTextures[i].clone();
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(repeatX, repeatY);
  t.needsUpdate = true;
  return t;
}
function material(g, mats) {
  const m = g.mat >= 0 ? mats[g.mat] : null;
  const rgba = m ? m.rgba : g.rgba;
  // texuniform: repetitions per metre -> scale by the (top face) size of the geom
  let rx = 1, ry = 1;
  if (m && m.tex >= 0) {
    rx = m.texrepeat[0]; ry = m.texrepeat[1];
    if (m.texuniform) { rx *= 2 * (g.size[0] || 1); ry *= 2 * (g.size[1] || 1); }
  }
  const key = `${g.mat}|${rgba.join(',')}|${rx.toFixed(3)},${ry.toFixed(3)}`;
  if (!matCache.has(key)) {
    const hint = pbrHints(m?.name, m);
    const color = new THREE.Color(rgba[0], rgba[1], rgba[2]).convertSRGBToLinear();
    const mat = new THREE.MeshStandardMaterial({
      color, roughness: hint.roughness, metalness: hint.metalness,
      transparent: rgba[3] < 1, opacity: rgba[3],
      map: m ? textureFor(m.tex, rx, ry) : null,
    });
    if (m && m.emission > 0) { mat.emissive = color.clone(); mat.emissiveIntensity = 1.5 * m.emission; }
    matCache.set(key, mat);
  }
  return matCache.get(key);
}
function b64(str, Type) {
  const bin = atob(str);
  const buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return new Type(buf.buffer);
}
function primitive(g) {
  const s = g.size;
  switch (g.type) {
    case 'box': return new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
    case 'sphere': return new THREE.SphereGeometry(s[0], 24, 16);
    case 'ellipsoid': return new THREE.SphereGeometry(1, 24, 16).scale(s[0], s[1], s[2]);
    case 'cylinder': return new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 32).rotateX(Math.PI / 2);
    case 'capsule': return new THREE.CapsuleGeometry(s[0], 2 * s[1], 8, 16).rotateX(Math.PI / 2);
    case 'plane': return new THREE.PlaneGeometry(2 * (s[0] || 10), 2 * (s[1] || 10));
  }
  return null;
}
function loadTexture(url) {
  return new Promise((resolve) => texLoader.load(url, (t) => {
    t.colorSpace = THREE.SRGBColorSpace;
    t.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
    resolve(t);
  }, undefined, () => resolve(null)));
}

// Deformable cloth: the server streams the coarse simulation vertices; the client
// smooths them with a fixed Loop-subdivision matrix W (fine = W * coarse) every frame.
function makeCloth(fx, mats) {
  const rows = b64(fx.w_rows, Uint32Array), cols = b64(fx.w_cols, Uint32Array), vals = b64(fx.w_vals, Float32Array);
  const geo = new THREE.BufferGeometry();
  const fine = new Float32Array(3 * fx.nfine);
  geo.setAttribute('position', new THREE.BufferAttribute(fine, 3).setUsage(THREE.DynamicDrawUsage));
  geo.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(3 * fx.nfine), 3).setUsage(THREE.DynamicDrawUsage));
  geo.setAttribute('uv', new THREE.BufferAttribute(b64(fx.uv, Float32Array), 2));
  geo.setIndex(new THREE.BufferAttribute(b64(fx.faces, Uint32Array), 1));
  const m = fx.mat >= 0 ? mats[fx.mat] : null;
  const fabric = (tex) => new THREE.MeshPhysicalMaterial({
    color: 0xffffff, map: tex, bumpMap: tex, bumpScale: 1.2,
    roughness: 0.92, metalness: 0.0,
    sheen: 0.6, sheenRoughness: 0.75, sheenColor: new THREE.Color(0.55, 0.62, 0.78),  // cotton fuzz
  });
  const frontMat = fabric(m ? textureFor(m.tex) : null);
  frontMat.side = THREE.FrontSide;
  // Inside of the garment (seen through the waist and hem openings): the reverse of denim
  // is the pale weft side, so reuse the weave for bump but tint it light.
  const backMat = fx.back_tex >= 0 ? fabric(textureFor(fx.back_tex)) : new THREE.MeshPhysicalMaterial({
    color: new THREE.Color(0.62, 0.68, 0.78), bumpMap: frontMat.map, bumpScale: 0.8,
    roughness: 0.95, metalness: 0.0, sheen: 0.4, sheenRoughness: 0.8, sheenColor: new THREE.Color(0.8, 0.84, 0.9),
  });
  backMat.side = THREE.BackSide;
  // the inside is never in front of the outside: bias it back in the depth test so the two
  // panels (a few mm apart) never flicker
  backMat.polygonOffset = true; backMat.polygonOffsetFactor = 2; backMat.polygonOffsetUnits = 4;
  const grp = new THREE.Group();
  for (const mat of [frontMat, backMat]) {
    const mesh = new THREE.Mesh(geo, mat);
    mesh.castShadow = mesh.receiveShadow = !LITE;
    mesh.frustumCulled = false;
    grp.add(mesh);
  }
  mjRoot.add(grp);
  return { grp, geo, rows, cols, vals, nvert: fx.nvert, coarse: new Float32Array(3 * fx.nvert), fine, dirty: false };
}
function updateCloth(c) {
  if (!c.dirty) return;
  c.dirty = false;
  const { rows, cols, vals, coarse, fine } = c;
  fine.fill(0);
  for (let k = 0; k < vals.length; k++) {
    const i = 3 * rows[k], j = 3 * cols[k], w = vals[k];
    fine[i] += w * coarse[j]; fine[i + 1] += w * coarse[j + 1]; fine[i + 2] += w * coarse[j + 2];
  }
  c.geo.attributes.position.needsUpdate = true;
  c.geo.computeVertexNormals();
}

async function loadScene() {
  const res = await fetch('/scene.json', { cache: 'no-store' });
  const js = await res.json();
  bodyObjs.forEach(o => mjRoot.remove(o));
  cloths.forEach(c => mjRoot.remove(c.grp));
  matCache.clear();
  sceneTextures = await Promise.all((js.textures || []).map(loadTexture));
  const mats = js.materials || {};
  bodyObjs = js.bodies.map(name => { const o = new THREE.Group(); o.name = name; mjRoot.add(o); return o; });
  const meshGeoms = js.meshes.map(me => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(b64(me.vert, Float32Array), 3));
    geo.setAttribute('normal', new THREE.BufferAttribute(b64(me.normal, Float32Array), 3));
    if (me.indexed) geo.setIndex(new THREE.BufferAttribute(b64(me.face, Uint32Array), 1));
    geo.computeBoundingSphere();
    return geo;
  });
  for (const g of js.geoms) {
    const geo = g.type === 'mesh' ? meshGeoms[g.mesh] : primitive(g);
    if (!geo) continue;
    const mesh = new THREE.Mesh(geo, material(g, mats));
    mesh.position.set(g.pos[0], g.pos[1], g.pos[2]);
    mesh.quaternion.set(g.quat[1], g.quat[2], g.quat[3], g.quat[0]);
    const big = g.type === 'plane' || /^(wall|skirt|ceiling)/.test(g.name || '');
    mesh.castShadow = !LITE && !big;
    mesh.receiveShadow = !LITE;
    bodyObjs[g.body].add(mesh);
  }
  cloths = (js.flexes || []).map(fx => makeCloth(fx, mats));
  window.__teleop = { scene, cloths, camera, orbit };        // debugging handle (browser console)
  document.getElementById('loading').style.display = 'none';
}

// Target markers (where the operator is commanding each EE)
function makeTarget(color) {
  const grp = new THREE.Group();
  grp.add(new THREE.Mesh(new THREE.SphereGeometry(0.012, 16, 12),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.8 })));
  grp.add(new THREE.AxesHelper(0.06));
  grp.visible = false;
  mjRoot.add(grp);
  return grp;
}
const targets = [makeTarget(0x33aaff), makeTarget(0xff9933)];

// ----------------------------------------------- laptop mirror of the VR user's view
// "Operator view": the laptop camera follows the headset wearer's head (smoothed).
// "Free orbit": mouse orbit. Operator view is chosen automatically when a headset is active.
const operatorHead = new THREE.Object3D();      // pose in MuJoCo frame, child of mjRoot
mjRoot.add(operatorHead);
let operatorLive = false;
let viewMode = 'auto';                           // 'auto' | 'operator' | 'free'
const _hp = new THREE.Vector3(), _hq = new THREE.Quaternion();
function updateMirrorCamera() {
  const follow = viewMode === 'operator' || (viewMode === 'auto' && operatorLive);
  orbit.enabled = !follow;
  if (!follow) { if (camera.fov !== 60) { camera.fov = 60; camera.updateProjectionMatrix(); } return; }
  operatorHead.updateMatrixWorld(true);
  operatorHead.matrixWorld.decompose(_hp, _hq, _s);
  camera.position.lerp(_hp, 0.5);
  camera.quaternion.slerp(_hq, 0.35);
  if (camera.fov !== 90) { camera.fov = 90; camera.updateProjectionMatrix(); }
}
const viewBtn = document.createElement('button');
function refreshViewBtn() {
  viewBtn.textContent = { auto: 'View: auto (follows headset)', operator: 'View: operator', free: 'View: free orbit' }[viewMode];
}
viewBtn.onclick = () => {
  viewMode = { auto: 'operator', operator: 'free', free: 'auto' }[viewMode];
  if (viewMode === 'free') { camera.position.set(0, 1.6, 0.12); orbit.target.set(0, 0.8, -0.75); orbit.update(); }
  refreshViewBtn();
};
refreshViewBtn();
document.getElementById('xrrow').appendChild(viewBtn);
window.addEventListener('keydown', (e) => {
  if (e.code === 'KeyV') viewBtn.onclick();
  if (e.code === 'KeyH') for (const id of ['hud', 'help'])      // hide/show overlays
    document.getElementById(id).style.display = document.getElementById(id).style.display === 'none' ? '' : 'none';
  if (e.code === 'KeyF') document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen();
});

// ---------------------------------------------------------------- network
let ws = null, wsOpen = false, status = {};
let sceneId = null, reloading = false;   // the server rebuilt its scene (e.g. another garment)
const connEl = document.getElementById('conn');
function connect() {
  ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => { wsOpen = true; connEl.className = 'ok'; };
  ws.onclose = () => { wsOpen = false; connEl.className = 'bad'; setTimeout(connect, 1000); };
  ws.onmessage = (ev) => {
    if (typeof ev.data === 'string') {
      status = JSON.parse(ev.data);
      if (status.scene_id !== undefined) {
        if (sceneId !== null && status.scene_id !== sceneId && !reloading) {
          reloading = true;
          loadScene().finally(() => { reloading = false; });
        }
        sceneId = status.scene_id;
      }
      updateHud();
      return;
    }
    const kind = new Uint8Array(ev.data, 0, 1)[0];
    if (kind === 2) { onCameraFrame(new Uint8Array(ev.data, 1, 1)[0], ev.data.slice(4)); return; }
    const f = new Float32Array(ev.data, 4);
    const flags = f[1], nb = f[2];
    if (nb !== bodyObjs.length) return;
    let k = 4;
    // robot poses: shown interpolated between the last two packets (smoothBodies), so the arms
    // move smoothly at the headset's frame rate however often the server sends
    const now = performance.now();
    if (!bodyPose.cur || bodyPose.cur.length !== 7 * nb) {
      bodyPose.prev = f.slice(k, k + 7 * nb); bodyPose.cur = bodyPose.prev.slice();
    } else {
      bodyPose.prev.set(bodyPose.cur); bodyPose.cur.set(f.subarray(k, k + 7 * nb));
      const gap = Math.min(Math.max(now - bodyPose.t, 4), 120);
      bodyPose.interval += 0.2 * (gap - bodyPose.interval);
    }
    bodyPose.t = now;
    k += 7 * nb;
    for (let a = 0; a < 2; a++, k += 7) {
      targets[a].position.set(f[k], f[k + 1], f[k + 2]);
      targets[a].quaternion.set(f[k + 4], f[k + 5], f[k + 6], f[k + 3]);
      targets[a].visible = (flags & (1 << a)) !== 0;
    }
    if (f.length >= k + 7) {           // operator (VR user) head pose, for the laptop mirror
      operatorLive = (flags & 8) !== 0;
      operatorHead.position.set(f[k], f[k + 1], f[k + 2]);
      operatorHead.quaternion.set(f[k + 4], f[k + 5], f[k + 6], f[k + 3]);
      k += 7;
    }
    const nCloth = f[3];               // cloth vertex positions (all garments, concatenated)
    if (nCloth > 0 && f.length >= k + 3 * nCloth) {
      for (const c of cloths) {
        c.coarse.set(f.subarray(k, k + 3 * c.nvert));
        c.dirty = true;
        k += 3 * c.nvert;
      }
    }
  };
}
function send(obj) { if (wsOpen) ws.send(JSON.stringify(obj)); }
function cmd(c) { send({ type: 'cmd', cmd: c }); }

// --------------------------------------------------------- camera screens
// Live MuJoCo camera renders (JPEG from the server) on screens floating beyond the table.
const CAM_LABELS = { left_gripper_wrist_cam: 'LEFT WRIST', right_gripper_wrist_cam: 'RIGHT WRIST',
  head_cam: 'HEAD', overhead: 'OVERHEAD', front: 'FRONT' };
const camScreens = [];            // index = server camera index
const camGroup = new THREE.Group();
world.add(camGroup);
const camDomRow = document.getElementById('cams');
function screenFor(i) {
  while (camScreens.length < i) makeScreen(camScreens.length);   // keep DOM/3D order = camera index
  return camScreens[i] || makeScreen(i);
}
function makeScreen(i) {
  const canvas = document.createElement('canvas');
  canvas.width = 320; canvas.height = 240;
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  const W = 0.44, H = 0.33;
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(W, H), new THREE.MeshBasicMaterial({ map: tex, toneMapped: false }));
  const frame = new THREE.Mesh(new THREE.PlaneGeometry(W + 0.02, H + 0.02), new THREE.MeshBasicMaterial({ color: 0x111111 }));
  frame.position.z = -0.002;
  const grp = new THREE.Group(); grp.add(frame); grp.add(mesh);
  camGroup.add(grp);
  const img = document.createElement('img'); camDomRow.appendChild(img);
  camScreens[i] = { canvas, tex, grp, img, busy: false };
  layoutScreens();
  return camScreens[i];
}
function layoutScreens() {
  // Arc of screens beyond the far edge of the table, facing you, slightly tilted down.
  const n = camScreens.length;
  camScreens.forEach((sc, i) => {
    if (!sc) return;
    const y = ((n - 1) / 2 - i) * 0.5;               // MuJoCo +y = your left; first camera on the left
    const p = new THREE.Vector3(1.1, y, 1.68).applyMatrix4(mjRoot.matrix);
    sc.grp.position.copy(p);
    const eye = OPERATOR_EYE_MJ.clone().applyMatrix4(mjRoot.matrix);
    const d = eye.sub(p);
    sc.grp.rotation.order = 'YXZ';
    sc.grp.rotation.set(-Math.atan2(d.y, Math.hypot(d.x, d.z)), Math.atan2(d.x, d.z), 0);
  });
}
function onCameraFrame(i, buf) {
  const sc = screenFor(i);
  if (sc.busy) return;            // drop frames if decoding is behind
  sc.busy = true;
  const blob = new Blob([buf], { type: 'image/jpeg' });
  createImageBitmap(blob).then(bmp => {
    const c = sc.canvas.getContext('2d');
    c.drawImage(bmp, 0, 0, sc.canvas.width, sc.canvas.height);
    const name = status.cams?.[i] || '';
    c.fillStyle = 'rgba(0,0,0,0.55)'; c.fillRect(0, 0, 140, 26);
    c.fillStyle = '#fff'; c.font = 'bold 16px system-ui, sans-serif';
    c.fillText(CAM_LABELS[name] || name, 8, 18);
    sc.tex.needsUpdate = true;
    bmp.close();
    if (!renderer.xr.isPresenting) {       // desktop thumbnails
      if (sc.img.src) URL.revokeObjectURL(sc.img.src);
      sc.img.src = URL.createObjectURL(blob);
    }
    sc.busy = false;
  }).catch(() => { sc.busy = false; });
}

// ----------------------------------------------------------------- HUDs
const statusEl = document.getElementById('status');
const btnRec = document.getElementById('btnRec');
btnRec.onclick = () => cmd('record_toggle');
document.getElementById('btnFail').onclick = () => cmd('save_fail');
document.getElementById('btnDiscard').onclick = () => cmd('discard');
document.getElementById('btnReset').onclick = () => cmd('reset');
window.addEventListener('keydown', (e) => {
  if (e.code === 'Space') { e.preventDefault(); cmd('record_toggle'); }
  if (e.code === 'KeyR') cmd('reset');
});

// In-VR panel on the left wrist
const hudCanvas = document.createElement('canvas');
hudCanvas.width = 512; hudCanvas.height = 384;
const hudTex = new THREE.CanvasTexture(hudCanvas);
hudTex.colorSpace = THREE.SRGBColorSpace;
const hudMesh = new THREE.Mesh(new THREE.PlaneGeometry(0.16, 0.12),
  new THREE.MeshBasicMaterial({ map: hudTex, transparent: true }));
hudMesh.position.set(0, 0.06, 0.03);
hudMesh.rotation.x = -0.9;

function updateHud() {
  const s = status;
  const stale = s.version !== PROTOCOL_VERSION;
  if (stale) s.message = 'VERSION MISMATCH: restart server + reload page';
  const rec = s.recording;
  const cov = s.cloth && s.cloth.coverage !== undefined ? `   folded: ${Math.round(100 * (1 - s.cloth.coverage))}% smaller footprint` : '';
  const hold = (h) => s.holding?.[h] ? ' (holding cloth)' : '';
  const g = s.game;
  const gameLine = !g ? '' : g.done
    ? `FOLDED in ${g.time.toFixed(0)} s` + (g.best !== null ? `   best ${g.best.toFixed(0)} s` : '') + '   (X = new game)'
    : `step ${g.step + 1}/${g.steps}: ${g.goal}   ${g.running ? g.time.toFixed(0) + ' s' : '(grip to start)'}` +
      (g.best !== null ? `   best ${g.best.toFixed(0)} s` : '');
  statusEl.textContent =
    (gameLine ? gameLine + '\n' : '') +
    `task: ${s.task_title || s.task || '-'}${cov}\n` +
    `${rec ? '● REC ' + s.rec_time.toFixed(1) + 's (' + s.frames + ' fr)' : 'idle'}\n` +
    `next episode: ${s.next_episode}   last: ${s.last_saved || '-'}\n` +
    `sim t=${s.sim_time}s  rtf=${s.rtf}  input=${s.input_fresh ? 'live' : 'none'}\n` +
    `engaged L:${s.engaged?.left ? 'Y' : '-'} R:${s.engaged?.right ? 'Y' : '-'}   ` +
    `gripper L:${Math.round(100 * (s.gripper?.left || 0))}%${hold('left')} R:${Math.round(100 * (s.gripper?.right || 0))}%${hold('right')}\n${s.message || ''}`;
  camGroup.visible = !!s.cams_on;
  camDomRow.style.display = s.cams_on ? 'flex' : 'none';
  btnRec.textContent = rec ? '■ Stop & save' : '● Record';
  btnRec.className = rec ? 'rec' : '';
  const c = hudCanvas.getContext('2d');
  c.clearRect(0, 0, 512, 384);
  c.fillStyle = 'rgba(15,17,22,0.85)'; c.beginPath(); c.roundRect(0, 0, 512, 384, 24); c.fill();
  if (g) {                     // the folding game: step, goal, clock
    const pct = Math.round(100 * (s.cloth?.coverage ?? 1));
    c.fillStyle = g.done ? '#5ee07a' : '#ffd24a';
    c.font = 'bold 34px system-ui, sans-serif';
    c.fillText(g.done ? `FOLDED  ${g.time.toFixed(0)} s` : `STEP ${g.step + 1}/${g.steps}   ${g.running ? g.time.toFixed(0) + ' s' : ''}`, 24, 300);
    c.fillStyle = '#eee'; c.font = '24px system-ui, sans-serif';
    c.fillText(g.done ? (g.best !== null ? `best ${g.best.toFixed(0)} s  -  X for a new game` : '') : g.goal.slice(0, 40), 24, 336);
    if (!g.done) {
      c.fillStyle = '#9ab';
      c.fillText(`footprint ${pct}% -> ${Math.round(100 * g.target)}%` + (g.best !== null ? `   best ${g.best.toFixed(0)} s` : ''), 24, 368);
    }
  }
  c.fillStyle = rec ? '#ff4040' : '#8a8f98';
  c.font = 'bold 54px system-ui, sans-serif';
  c.fillText(rec ? `● REC ${s.rec_time.toFixed(1)}s` : '○ idle', 24, 72);
  c.fillStyle = '#ddd'; c.font = '30px system-ui, sans-serif';
  c.fillText(`episode ${s.next_episode}   ${s.last_saved ? 'last: ' + s.last_saved.replace('.hdf5', '') : ''}`, 24, 128);
  const gp = (h) => `${s.engaged?.[h] ? '■' : '□'} grip ${Math.round(100 * (s.gripper?.[h] || 0))}%${s.holding?.[h] ? '✋' : ''}`;
  c.fillText(`L ${gp('left')}   R ${gp('right')}`, 24, 176);
  c.fillStyle = '#9ab'; c.font = '24px system-ui, sans-serif';
  c.fillText((s.message || '').slice(0, 38), 24, 226);
  hudTex.needsUpdate = true;
}

// ------------------------------------------------------------- controllers
const factory = new XRControllerModelFactory();
const grips = {};            // handedness -> grip Object3D
for (let i = 0; i < 2; i++) {
  const grip = renderer.xr.getControllerGrip(i);
  grip.add(factory.createControllerModel(grip));
  grip.addEventListener('connected', (e) => {
    const hand = e.data.handedness;
    grip.userData.hand = hand;
    grips[hand] = grip;
    if (hand === 'left') grip.add(hudMesh);
  });
  grip.addEventListener('disconnected', () => {
    const hand = grip.userData.hand;
    if (hand && grips[hand] === grip) delete grips[hand];
    if (hudMesh.parent === grip) grip.remove(hudMesh);
  });
  scene.add(grip);
}

const prevButtons = { left: [], right: [] };
const invRoot = new THREE.Matrix4(), rel = new THREE.Matrix4();
const _p = new THREE.Vector3(), _q = new THREE.Quaternion(), _s = new THREE.Vector3();
function mjPose(obj) {
  invRoot.copy(mjRoot.matrixWorld).invert();
  rel.multiplyMatrices(invRoot, obj.matrixWorld);
  rel.decompose(_p, _q, _s);
  return [_p.x, _p.y, _p.z, _q.w, _q.x, _q.y, _q.z];
}
function pulse(src, strength = 0.6, ms = 40) {
  try { src.gamepad?.hapticActuators?.[0]?.pulse(strength, ms); } catch (_) {}
}
// Put your eyes exactly at the robot's head (position + height) facing the table.
function recenter() {
  const xrCam = renderer.xr.getCamera();
  const head = new THREE.Vector3().setFromMatrixPosition(xrCam.matrixWorld);
  const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(xrCam.getWorldQuaternion(new THREE.Quaternion()));
  const yaw = Math.atan2(-fwd.x, -fwd.z);
  world.rotation.set(0, yaw, 0);
  const off = eyeMJ().applyMatrix4(mjRoot.matrix).applyAxisAngle(new THREE.Vector3(0, 1, 0), yaw);
  world.position.set(head.x - off.x, head.y - off.y, head.z - off.z);
  placeLights();
  worldEpoch++;
}
// Slide the viewpoint by d (MuJoCo frame): the world moves the opposite way.
function moveView(d) {
  const before = viewOffset.clone();
  viewOffset.add(d).clamp(new THREE.Vector3(-1.0, -0.8, -0.6), new THREE.Vector3(0.8, 0.8, 0.8));
  const w = viewOffset.clone().sub(before).applyMatrix4(mjRoot.matrix).applyAxisAngle(new THREE.Vector3(0, 1, 0), world.rotation.y);
  world.position.sub(w);
  placeLights();
  worldEpoch++;              // re-anchor engaged arms so they don't jump
}
let viewSaveT = 0;
let recenterIn = 0;          // frames until auto-recenter after entering VR
renderer.xr.addEventListener('sessionstart', () => { recenterIn = 30; });
renderer.xr.addEventListener('sessionend', () => { defaultPlacement(); placeLights(); });

let lastT = performance.now();
function handleXRInput() {
  const session = renderer.xr.getSession();
  if (!session) return;
  const now = performance.now();
  const dt = Math.min((now - lastT) / 1000, 0.1);
  lastT = now;
  if (recenterIn > 0 && --recenterIn === 0) recenter();
  scene.updateMatrixWorld();
  const msg = { type: 'input', t: now / 1000, left: null, right: null, head: mjPose(renderer.xr.getCamera()) };
  let anyEngaged = false;
  for (const src of session.inputSources) {
    const hand = src.handedness;
    if (!src.gamepad || !grips[hand]) continue;
    const b = src.gamepad.buttons.map(x => x.pressed);
    const v = src.gamepad.buttons.map(x => x.value);
    const grip = v[1] ?? 0;
    const ax = src.gamepad.axes;
    let stick = ax.length >= 4 ? [ax[2], ax[3]] : [ax[0] ?? 0, ax[1] ?? 0];
    if (grip > 0.5) anyEngaged = true;
    // Trigger held: the stick moves your viewpoint instead of the gripper.
    // Left: forward/back + sideways. Right: up/down.
    const trig = (v[0] ?? 0) > 0.5;
    if (trig) {
      const dz = (x) => Math.abs(x) > 0.15 ? x : 0;
      const sp = 0.35 * dt;          // m/s at full deflection
      const d = hand === 'left'
        ? new THREE.Vector3(-dz(stick[1]) * sp, -dz(stick[0]) * sp, 0)
        : new THREE.Vector3(0, 0, -dz(stick[1]) * sp);
      if (d.lengthSq() > 0) { moveView(d); viewSaveT = now; }
      stick = [0, 0];
    }
    // stick up/down drives the gripper: UP = close, DOWN = open. WebXR reports up as negative
    // y and the server closes on positive y, so y is flipped here.
    stick = [stick[0], -stick[1]];
    msg[hand] = { pose: mjPose(grips[hand]), grip, stick, buttons: b.map(x => x ? 1 : 0), epoch: worldEpoch };
    const pb = prevButtons[hand];
    const edge = (i) => b[i] && !pb[i];
    if ((grip > 0.6) !== ((pb.gripV ?? 0) > 0.6)) pulse(src, 0.3, 25);
    if (hand === 'right' && edge(4)) { cmd('record_toggle'); pulse(src, 1.0, 120); }
    if (hand === 'right' && edge(5)) { cmd('discard'); pulse(src, 0.8, 60); }
    if (hand === 'left' && edge(4)) { cmd('reset'); pulse(src, 0.8, 60); }
    if (hand === 'left' && edge(5) && !anyEngaged) {
      if (trig) { viewOffset.set(0, 0, 0); saveViewOffset(); }     // left trigger + Y: default viewpoint
      recenter(); pulse(src, 0.5, 40);
    }
    if (edge(3)) { cmd('cams_toggle'); pulse(src, 0.5, 40); }     // click either stick
    prevButtons[hand] = Object.assign(b, { gripV: grip });
  }
  if (viewSaveT && now - viewSaveT > 500) { saveViewOffset(); viewSaveT = 0; }
  send(msg);
}

// -------------------------------------------------------------------- loop
const bodyPose = { prev: null, cur: null, t: 0, interval: 11 };
const _qa = new THREE.Quaternion(), _qb = new THREE.Quaternion();
function smoothBodies() {
  const { prev, cur } = bodyPose;
  if (!cur || cur.length !== 7 * bodyObjs.length) return;
  // one packet interval behind the newest pose: always between two real poses
  const a = Math.min(1, (performance.now() - bodyPose.t) / bodyPose.interval);
  for (let i = 0, k = 0; i < bodyObjs.length; i++, k += 7) {
    const o = bodyObjs[i];
    o.position.set(prev[k] + a * (cur[k] - prev[k]), prev[k + 1] + a * (cur[k + 1] - prev[k + 1]),
      prev[k + 2] + a * (cur[k + 2] - prev[k + 2]));
    _qa.set(prev[k + 4], prev[k + 5], prev[k + 6], prev[k + 3]);
    _qb.set(cur[k + 4], cur[k + 5], cur[k + 6], cur[k + 3]);
    o.quaternion.slerpQuaternions(_qa, _qb, a);
  }
}
renderer.setAnimationLoop(() => {
  handleXRInput();
  smoothBodies();
  for (const c of cloths) updateCloth(c);
  if (!renderer.xr.isPresenting) updateMirrorCamera();
  renderer.render(scene, camera);
});
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

loadScene().then(connect).catch(err => {
  document.getElementById('loading').textContent = 'Failed to load scene: ' + err;
  console.error(err);
});
