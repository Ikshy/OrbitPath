/**
 * ═══════════════════════════════════════════════════════════════
 * OrbitPath — main.js  v3.0
 * ═══════════════════════════════════════════════════════════════
 * Full Three.js Space Situational Awareness dashboard.
 *
 * Architecture:
 *   GlobeRenderer  — Three.js scene, textured Earth, atmosphere,
 *                    satellite meshes + trails, orbit lines, arcs
 *   APIClient      — REST calls to Flask backend
 *   UIManager      — DOM updates, clock, panels, tooltip
 *   App            — Orchestrator, polling, event wiring
 *
 * Earth textures: NASA Blue Marble (public domain, via NASA servers)
 * ═══════════════════════════════════════════════════════════════
 */

"use strict";

/* ═══════════════════════════════════════════════════════ CONFIG */
const CFG = {
  API_BASE:         "http://localhost:5000",
  POLL_SATS_MS:     8_000,
  POLL_COLL_MS:     30_000,
  POLL_ANOM_MS:     60_000,
  DEFAULT_GROUP:    "stations",
  LIMIT:            100,
  EARTH_RADIUS:     1.0,           // Three.js unit
  SAT_RADIUS:       0.010,
  TRAIL_LENGTH:     40,            // positions per trail
  ORBIT_STEPS:      90,
  CAM_START_Z:      2.85,
};

const GROUP_COLORS = {
  stations: 0x00ffcc,
  active:   0x00d4ff,
  starlink: 0x7755ff,
  debris:   0xff6600,
};

const RISK_COLORS = {
  CRITICAL: 0xff1744,
  HIGH:     0xff6600,
  MEDIUM:   0xffcc00,
  LOW:      0x00e676,
};

/* ═══════════════════════════════════════════════════ API CLIENT */
class APIClient {
  constructor(base) { this.base = base; }

  async _get(path, params = {}) {
    const url = new URL(this.base + path);
    Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, String(v)));
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status} — ${path}`);
    const json = await res.json();
    if (!json.success) throw new Error(json.error || "API error");
    return json;
  }

  satellites(group, limit) { return this._get("/satellites", { group, limit }); }
  orbit(id, group, steps)  { return this._get(`/orbits/${id}`, { group, steps }); }
  collisions(group, limit) { return this._get("/collision-risk", { group, limit }); }
  anomalies(group, limit)  { return this._get("/anomaly-report", { group, limit }); }
  health()                 { return this._get("/health"); }
}

/* ══════════════════════════════════════════════ GLOBE RENDERER */
class GlobeRenderer {
  constructor(canvas) {
    this.canvas = canvas;

    // Three.js core
    this.scene    = null;
    this.cam      = null;
    this.renderer = null;

    // Earth objects
    this.earth = null;
    this.atm   = null;
    this.clouds = null;
    this.grid  = null;
    this.nightLights = null;

    // Groups (rotate together when user drags)
    this._worldGroup = null;   // earth + grid + atm
    this._satGroup   = null;   // satellite meshes
    this._trailGroup = null;   // trail lines
    this._orbitGroup = null;   // orbit path lines
    this._conjGroup  = null;   // conjunction arcs

    // State
    this._sats       = [];     // { mesh, trail, history[], data }
    this._orbLines   = [];
    this._conjLines  = [];
    this._drag       = false;
    this._prev       = { x: 0, y: 0 };
    this._autoRotate = true;
    this._mouseNDC   = new THREE.Vector2();
    this._ray        = null;
    this._showTrails = true;
    this._showOrbits = true;
    this._showConj   = true;
    this._showLabels = false;   // kept for future CSS2DRenderer extension

    // Callbacks
    this.onHoverSat  = null;
    this.onClickSat  = null;

    // Clock for smooth animation
    this._clock = new THREE.Clock();
  }

  /* ── Initialise Three.js scene ──────────────────────────── */
  init() {
    const W = this.canvas.clientWidth;
    const H = this.canvas.clientHeight;

    /* Scene */
    this.scene = new THREE.Scene();

    /* Camera */
    this.cam = new THREE.PerspectiveCamera(42, W / H, 0.005, 300);
    this.cam.position.z = CFG.CAM_START_Z;

    /* Renderer */
    this.renderer = new THREE.WebGLRenderer({
      canvas:    this.canvas,
      antialias: true,
      alpha:     true,
    });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setSize(W, H);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;

    /* Raycaster */
    this._ray = new THREE.Raycaster();
    this._ray.params.Points = { threshold: 0.05 };

    /* Groups */
    this._worldGroup = new THREE.Group();
    this._satGroup   = new THREE.Group();
    this._trailGroup = new THREE.Group();
    this._orbitGroup = new THREE.Group();
    this._conjGroup  = new THREE.Group();

    this.scene.add(
      this._worldGroup,
      this._satGroup,
      this._trailGroup,
      this._orbitGroup,
      this._conjGroup
    );

    this._buildLights();
    this._buildEarth();
    this._buildAtmosphere();
    this._buildClouds();
    this._buildStarfield();
    this._buildOrbitDecoration();
    this._attachEvents();
    this._loop();
  }

  /* ── Lighting ────────────────────────────────────────────── */
  _buildLights() {
    /* Sunlight */
    const sun = new THREE.DirectionalLight(0xfff5e0, 2.2);
    sun.position.set(8, 4, 6);
    this.scene.add(sun);

    /* Ambient (space dark) */
    this.scene.add(new THREE.AmbientLight(0x060c18, 1.2));

    /* Blue atmospheric rim */
    const rim = new THREE.PointLight(0x1a5fff, 0.7, 12);
    rim.position.set(-6, 2, -5);
    this.scene.add(rim);

    /* Subtle fill from below */
    const fill = new THREE.PointLight(0x002244, 0.4, 8);
    fill.position.set(0, -5, 0);
    this.scene.add(fill);
  }

  /* ── Earth (textured) ────────────────────────────────────── */
  _buildEarth() {
    const loader = new THREE.TextureLoader();
    const geo    = new THREE.SphereGeometry(CFG.EARTH_RADIUS, 80, 80);

    /*
     * NASA Blue Marble textures (public domain).
     * Day map → phong diffuse
     * Specular map → ocean shine
     * Normal/bump map → terrain relief
     *
     * NOTE: These URLs work when the browser can reach the internet.
     * Fallback procedural material is used if textures fail to load.
     */
    const dayURL  = "https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg";
    const specURL = "https://unpkg.com/three-globe/example/img/earth-water.png";
    const nightURL= "https://unpkg.com/three-globe/example/img/earth-night.jpg";

    // Fallback procedural material (used immediately; replaced when textures load)
    this.earth = new THREE.Mesh(geo, this._proceduralEarthMat());
    this._worldGroup.add(this.earth);

    // Load textures asynchronously
    loader.load(dayURL, (dayTex) => {
      dayTex.colorSpace = THREE.SRGBColorSpace || THREE.sRGBEncoding;

      loader.load(specURL, (specTex) => {
        this.earth.material = new THREE.MeshPhongMaterial({
          map:          dayTex,
          specularMap:  specTex,
          specular:     new THREE.Color(0x336677),
          shininess:    55,
          bumpScale:    0.012,
        });
      }, undefined, () => {
        // specular failed, just use day texture
        this.earth.material = new THREE.MeshPhongMaterial({
          map:       dayTex,
          specular:  new THREE.Color(0x224455),
          shininess: 35,
        });
      });
    }, undefined, () => {
      // Day texture failed — keep procedural
      console.warn("Earth texture load failed; using procedural fallback.");
    });

    /* Night-lights overlay (additive blend) */
    loader.load(nightURL, (nightTex) => {
      const nGeo = new THREE.SphereGeometry(CFG.EARTH_RADIUS + 0.001, 80, 80);
      const nMat = new THREE.MeshBasicMaterial({
        map:         nightTex,
        blending:    THREE.AdditiveBlending,
        transparent: true,
        opacity:     0.45,
        depthWrite:  false,
      });
      this.nightLights = new THREE.Mesh(nGeo, nMat);
      this._worldGroup.add(this.nightLights);
    });

    /* Lat/lon grid overlay */
    const gGeo = new THREE.SphereGeometry(CFG.EARTH_RADIUS + 0.003, 40, 20);
    const gMat = new THREE.MeshBasicMaterial({
      color:       0x003355,
      wireframe:   true,
      transparent: true,
      opacity:     0.10,
    });
    this.grid = new THREE.Mesh(gGeo, gMat);
    this._worldGroup.add(this.grid);

    /* Pole axis */
    const pGeo = new THREE.CylinderGeometry(0.002, 0.002, 2.5, 6);
    const pMat = new THREE.MeshBasicMaterial({ color: 0x00d4ff, transparent: true, opacity: 0.15 });
    this._worldGroup.add(new THREE.Mesh(pGeo, pMat));
  }

  /* Procedural Earth fallback */
  _proceduralEarthMat() {
    return new THREE.MeshPhongMaterial({
      color:     0x0d4060,
      emissive:  0x021020,
      specular:  0x225566,
      shininess: 55,
    });
  }

  /* ── Atmosphere (Fresnel-style glow) ─────────────────────── */
  _buildAtmosphere() {
    /* Inner glow */
    const aGeo = new THREE.SphereGeometry(CFG.EARTH_RADIUS * 1.04, 48, 48);
    const aMat = new THREE.MeshPhongMaterial({
      color:       0x2288ff,
      transparent: true,
      opacity:     0.055,
      side:        THREE.FrontSide,
      depthWrite:  false,
    });
    this.atm = new THREE.Mesh(aGeo, aMat);
    this._worldGroup.add(this.atm);

    /* Outer halo (backside) */
    const hGeo = new THREE.SphereGeometry(CFG.EARTH_RADIUS * 1.09, 48, 48);
    const hMat = new THREE.MeshBasicMaterial({
      color:       0x0044bb,
      transparent: true,
      opacity:     0.025,
      side:        THREE.BackSide,
      depthWrite:  false,
    });
    this._worldGroup.add(new THREE.Mesh(hGeo, hMat));

    /* Atmospheric ring shader — thin glowing edge */
    const rGeo = new THREE.SphereGeometry(CFG.EARTH_RADIUS * 1.015, 64, 64);
    const rMat = new THREE.ShaderMaterial({
      vertexShader: `
        varying vec3 vNormal;
        varying vec3 vPosition;
        void main() {
          vNormal   = normalize(normalMatrix * normal);
          vPosition = (modelViewMatrix * vec4(position, 1.0)).xyz;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        varying vec3 vNormal;
        varying vec3 vPosition;
        void main() {
          float rim   = 1.0 - abs(dot(normalize(vNormal), normalize(-vPosition)));
          float power = pow(rim, 3.5);
          vec3  col   = mix(vec3(0.0, 0.35, 1.0), vec3(0.0, 0.85, 1.0), power);
          gl_FragColor = vec4(col, power * 0.55);
        }
      `,
      transparent: true,
      side:        THREE.FrontSide,
      depthWrite:  false,
      blending:    THREE.AdditiveBlending,
    });
    this._worldGroup.add(new THREE.Mesh(rGeo, rMat));
  }

  /* ── Cloud layer ─────────────────────────────────────────── */
  _buildClouds() {
    const loader  = new THREE.TextureLoader();
    const cloudURL = "https://unpkg.com/three-globe/example/img/earth-clouds.png";

    loader.load(cloudURL, (tex) => {
      const cGeo = new THREE.SphereGeometry(CFG.EARTH_RADIUS * 1.006, 64, 64);
      const cMat = new THREE.MeshPhongMaterial({
        map:         tex,
        transparent: true,
        opacity:     0.38,
        depthWrite:  false,
      });
      this.clouds = new THREE.Mesh(cGeo, cMat);
      this._worldGroup.add(this.clouds);
    });
  }

  /* ── Procedural starfield ────────────────────────────────── */
  _buildStarfield() {
    const N   = 6000;
    const pos = new Float32Array(N * 3);
    const col = new Float32Array(N * 3);
    const sz  = new Float32Array(N);

    for (let i = 0; i < N; i++) {
      const r   = 30 + Math.random() * 40;
      const th  = Math.random() * Math.PI * 2;
      const ph  = Math.acos(2 * Math.random() - 1);
      pos[i*3]   = r * Math.sin(ph) * Math.cos(th);
      pos[i*3+1] = r * Math.sin(ph) * Math.sin(th);
      pos[i*3+2] = r * Math.cos(ph);

      const t   = Math.random();
      col[i*3]   = 0.80 + t * 0.20;
      col[i*3+1] = 0.88 + t * 0.12;
      col[i*3+2] = 1.0;
      sz[i] = 0.3 + Math.random() * 1.8;
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("color",    new THREE.BufferAttribute(col, 3));
    geo.setAttribute("size",     new THREE.BufferAttribute(sz,  1));

    const mat = new THREE.PointsMaterial({
      size:          0.055,
      vertexColors:  true,
      transparent:   true,
      opacity:       0.88,
      sizeAttenuation: true,
      depthWrite:    false,
    });

    this.scene.add(new THREE.Points(geo, mat));
  }

  /* ── Decorative orbital regime rings ─────────────────────── */
  _buildOrbitDecoration() {
    [
      { r: CFG.EARTH_RADIUS * 1.12, op: 0.12, col: 0x003366 },
      { r: CFG.EARTH_RADIUS * 1.25, op: 0.07, col: 0x002244 },
    ].forEach(({ r, op, col }) => {
      const geo = new THREE.RingGeometry(r - 0.001, r + 0.002, 160);
      const mat = new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: op, side: THREE.DoubleSide, depthWrite: false });
      const ring = new THREE.Mesh(geo, mat);
      ring.rotation.x = Math.PI / 2 + 0.18;
      this.scene.add(ring);
    });
  }

  /* ═══════════════════════════════════ SATELLITE MANAGEMENT */

  /**
   * Render all satellites as glowing spheres.
   * Each satellite gets a trail buffer (ring of past positions).
   */
  addSatellites(sats, group) {
    this.clearSatellites();

    const color  = GROUP_COLORS[group] || GROUP_COLORS.active;
    const satGeo = new THREE.SphereGeometry(CFG.SAT_RADIUS, 7, 7);

    sats.forEach(s => {
      /* Main body */
      const mat  = new THREE.MeshBasicMaterial({ color });
      const mesh = new THREE.Mesh(satGeo, mat);
      const pos  = this._lla2xyz(s.latitude, s.longitude, s.altitude);
      mesh.position.set(pos.x, pos.y, pos.z);
      mesh.userData = s;

      /* Glow sprite */
      const spMat = new THREE.SpriteMaterial({
        color, transparent: true, opacity: 0.30, depthWrite: false,
      });
      const sp = new THREE.Sprite(spMat);
      sp.scale.setScalar(CFG.SAT_RADIUS * 5.5);
      mesh.add(sp);

      /* Trail — initially empty; filled as satellite moves */
      const trailGeo  = new THREE.BufferGeometry();
      const trailPos  = new Float32Array(CFG.TRAIL_LENGTH * 3);
      trailGeo.setAttribute("position", new THREE.BufferAttribute(trailPos, 3));
      trailGeo.setDrawRange(0, 0);

      const trailMat = new THREE.LineBasicMaterial({
        color, transparent: true, opacity: 0.50, linewidth: 1, depthWrite: false,
      });
      const trail = new THREE.Line(trailGeo, trailMat);

      this._satGroup.add(mesh);
      this._trailGroup.add(trail);

      this._sats.push({
        mesh,
        trail,
        history: [ { ...pos } ],  // ring buffer of ECI-equivalent positions
        data: s,
      });
    });
  }

  /**
   * Update satellite positions on each API poll.
   * Appends to history for trail rendering.
   */
  updateSatPositions(freshSats) {
    const map = {};
    freshSats.forEach(s => { map[s.norad_id] = s; });

    this._sats.forEach(entry => {
      const fresh = map[entry.data.norad_id];
      if (!fresh) return;

      const pos = this._lla2xyz(fresh.latitude, fresh.longitude, fresh.altitude);
      entry.mesh.position.set(pos.x, pos.y, pos.z);
      entry.data = fresh;
      entry.mesh.userData = fresh;

      /* Trail history */
      entry.history.push({ ...pos });
      if (entry.history.length > CFG.TRAIL_LENGTH) {
        entry.history.shift();
      }

      if (this._showTrails) {
        this._updateTrailGeometry(entry);
      }
    });
  }

  _updateTrailGeometry(entry) {
    const hist = entry.history;
    const arr  = entry.trail.geometry.attributes.position.array;
    for (let i = 0; i < hist.length; i++) {
      arr[i * 3]     = hist[i].x;
      arr[i * 3 + 1] = hist[i].y;
      arr[i * 3 + 2] = hist[i].z;
    }
    entry.trail.geometry.attributes.position.needsUpdate = true;
    entry.trail.geometry.setDrawRange(0, hist.length);

    // Fade opacity toward tail end — update material alpha based on history fill
    const fillRatio = hist.length / CFG.TRAIL_LENGTH;
    entry.trail.material.opacity = this._showTrails ? 0.35 + fillRatio * 0.20 : 0;
  }

  clearSatellites() {
    this._sats.forEach(({ mesh, trail }) => {
      this._satGroup.remove(mesh);
      this._trailGroup.remove(trail);
    });
    this._sats = [];
  }

  /* Highlight anomalous satellites with amber colour */
  flagAnomalousSats(noradIds) {
    const flagged = new Set(noradIds);
    this._sats.forEach(({ mesh, data }) => {
      mesh.material.color.set(flagged.has(data.norad_id) ? 0xffaa00 : (GROUP_COLORS[this._currentGroup] || GROUP_COLORS.active));
    });
  }

  /* ═══════════════════════════════════════ ORBIT TRACK LINES */
  addOrbitTrack(track, group) {
    this.clearOrbitLines();
    if (!track || track.length < 2) return;

    const color  = GROUP_COLORS[group] || 0x00d4ff;
    const points = track.map(p => {
      const v = this._lla2xyz(p.lat, p.lon, p.alt);
      return new THREE.Vector3(v.x, v.y, v.z);
    });

    /* Main line */
    const geo  = new THREE.BufferGeometry().setFromPoints(points);
    const mat  = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.60, depthWrite: false });
    const line = new THREE.Line(geo, mat);
    this._orbitGroup.add(line);
    this._orbLines.push(line);

    /* Glow duplicate */
    const gMat  = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.18, depthWrite: false });
    const gLine = new THREE.Line(geo.clone(), gMat);
    this._orbitGroup.add(gLine);
    this._orbLines.push(gLine);
  }

  clearOrbitLines() {
    this._orbLines.forEach(l => this._orbitGroup.remove(l));
    this._orbLines = [];
  }

  /* ═══════════════════════════════════ CONJUNCTION ARCS */
  showConjArcs(events, satData) {
    this.clearConjLines();
    if (!this._showConj) return;

    const map = {};
    satData.forEach(s => { map[s.norad_id] = s; });

    events.slice(0, 8).forEach(evt => {
      const s1 = map[evt.sat1_id], s2 = map[evt.sat2_id];
      if (!s1 || !s2) return;

      const p1 = this._lla2xyz(s1.latitude, s1.longitude, s1.altitude);
      const p2 = this._lla2xyz(s2.latitude, s2.longitude, s2.altitude);
      const v1 = new THREE.Vector3(p1.x, p1.y, p1.z);
      const v2 = new THREE.Vector3(p2.x, p2.y, p2.z);

      /* Quadratic Bezier arc lifted above Earth */
      const mid = v1.clone().add(v2).multiplyScalar(0.5);
      mid.normalize().multiplyScalar(mid.length() * 1.14);

      const curve  = new THREE.QuadraticBezierCurve3(v1, mid, v2);
      const pts    = curve.getPoints(50);
      const geo    = new THREE.BufferGeometry().setFromPoints(pts);
      const color  = RISK_COLORS[evt.risk_level] || 0xffffff;
      const mat    = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.78, depthWrite: false });
      const line   = new THREE.Line(geo, mat);
      this._conjGroup.add(line);
      this._conjLines.push(line);

      /* Endpoint pulsar spheres */
      [v1, v2].forEach(v => {
        const pGeo = new THREE.SphereGeometry(0.018, 8, 8);
        const pMat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.7 });
        const pm   = new THREE.Mesh(pGeo, pMat);
        pm.position.copy(v);
        this._conjGroup.add(pm);
        this._conjLines.push(pm);
      });
    });
  }

  clearConjLines() {
    this._conjLines.forEach(l => this._conjGroup.remove(l));
    this._conjLines = [];
  }

  /* ═══════════════════════════════════════ TOGGLE HELPERS */
  setShowTrails(v) {
    this._showTrails = v;
    this._sats.forEach(e => { e.trail.material.opacity = v ? 0.50 : 0; });
  }

  setShowOrbits(v) {
    this._showOrbits = v;
    this._orbLines.forEach(l => { if (l.material) l.material.opacity = v ? 0.60 : 0; });
  }

  setShowConj(v) {
    this._showConj = v;
    if (!v) this.clearConjLines();
  }

  /* ═════════════════════════════════════ COORDINATE TOOLS */
  _lla2xyz(lat, lon, altKm) {
    const R   = CFG.EARTH_RADIUS + (altKm / 6371.0) * CFG.EARTH_RADIUS;
    const phi = THREE.MathUtils.degToRad(90 - lat);
    const tht = THREE.MathUtils.degToRad(lon + 180);
    return {
      x:  R * Math.sin(phi) * Math.cos(tht),
      y:  R * Math.cos(phi),
      z:  R * Math.sin(phi) * Math.sin(tht),
    };
  }

  /* ═══════════════════════════════════════════ GLOBE EVENTS */
  _rotateWorld(dx, dy) {
    const groups = [
      this._worldGroup, this._satGroup, this._trailGroup,
      this._orbitGroup, this._conjGroup,
    ];
    groups.forEach(g => { g.rotation.y += dx; g.rotation.x += dy; });
  }

  _attachEvents() {
    const c = this.canvas;

    c.addEventListener("mousedown", e => {
      this._drag = true;
      this._prev = { x: e.clientX, y: e.clientY };
      this._autoRotate = false;
    });

    window.addEventListener("mouseup", () => {
      this._drag = false;
      setTimeout(() => { this._autoRotate = true; }, 4000);
    });

    window.addEventListener("mousemove", e => {
      if (this._drag) {
        const dx = (e.clientX - this._prev.x) * 0.005;
        const dy = (e.clientY - this._prev.y) * 0.005;
        this._rotateWorld(dx, dy);
        this._prev = { x: e.clientX, y: e.clientY };
      }

      /* Hover detection (NDC) */
      const r = c.getBoundingClientRect();
      this._mouseNDC.x =  ((e.clientX - r.left) / r.width)  * 2 - 1;
      this._mouseNDC.y = -((e.clientY - r.top)  / r.height) * 2 + 1;

      if (this.onHoverSat) {
        this._ray.setFromCamera(this._mouseNDC, this.cam);
        const meshes = this._sats.map(s => s.mesh);
        const hits   = this._ray.intersectObjects(meshes, true);
        this.onHoverSat(hits.length ? hits[0].object.userData : null, e.clientX, e.clientY);
      }
    });

    c.addEventListener("click", () => {
      this._ray.setFromCamera(this._mouseNDC, this.cam);
      const hits = this._ray.intersectObjects(this._sats.map(s => s.mesh), true);
      if (hits.length && this.onClickSat) this.onClickSat(hits[0].object.userData);
    });

    c.addEventListener("wheel", e => {
      this.cam.position.z = Math.max(1.4, Math.min(7.0, this.cam.position.z + e.deltaY * 0.002));
    }, { passive: true });

    window.addEventListener("resize", () => {
      const W = c.clientWidth, H = c.clientHeight;
      this.cam.aspect = W / H;
      this.cam.updateProjectionMatrix();
      this.renderer.setSize(W, H);
    });
  }

  /* ════════════════════════════════════════ RENDER LOOP */
  _loop() {
    requestAnimationFrame(() => this._loop());

    const dt = this._clock.getDelta();

    /* Slow auto-rotation */
    if (this._autoRotate) this._rotateWorld(0.00065, 0);

    /* Cloud layer drifts slightly faster */
    if (this.clouds) this.clouds.rotation.y += 0.00012;

    this.renderer.render(this.scene, this.cam);
  }
}

/* ═══════════════════════════════════════════════ UI MANAGER */
const UI = {
  /* Panel collapse / expand (called from onclick in HTML) */
  toggle(hdr) {
    const body = hdr.nextElementSibling;
    if (body) body.classList.toggle("open");
  },

  /* UTC clock */
  _startClock() {
    const el  = document.getElementById("clock");
    const pad = n => String(n).padStart(2, "0");
    setInterval(() => {
      const n = new Date();
      if (el) el.textContent = `${pad(n.getUTCHours())}:${pad(n.getUTCMinutes())}:${pad(n.getUTCSeconds())}`;
    }, 1000);
  },

  setMetric(id, val, cls) {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = val;
    if (cls) el.className = `mc-value ${cls}`;
  },

  setBadge(id, val, cls) {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = val;
    if (cls) el.className = `p-badge ${cls}`;
  },

  renderSatList(sats, onSelect) {
    const el = document.getElementById("sat-list");
    if (!el) return;
    el.innerHTML = "";
    if (!sats.length) { el.innerHTML = '<div class="empty">No satellites loaded</div>'; return; }

    sats.forEach(s => {
      const row = document.createElement("div");
      row.className = "sat-row";
      row.innerHTML = `
        <div class="s-dot" style="background:var(--sat-active);color:var(--sat-active);"></div>
        <span class="s-name">${esc(s.name)}</span>
        <span class="s-alt">${s.altitude?.toFixed(0) ?? "--"} km</span>
      `;
      row.addEventListener("click", () => {
        document.querySelectorAll(".sat-row").forEach(r => r.classList.remove("active"));
        row.classList.add("active");
        onSelect(s);
      });
      el.appendChild(row);
    });
  },

  showSatDetail(s) {
    const panel = document.getElementById("sel-panel");
    const body  = document.getElementById("sel-body");
    if (!panel || !body) return;
    panel.style.display = "block";

    const rows = [
      ["NORAD ID",   s.norad_id],
      ["Name",       s.name],
      ["Latitude",   `${s.latitude?.toFixed(4)}°`],
      ["Longitude",  `${s.longitude?.toFixed(4)}°`],
      ["Altitude",   `${s.altitude?.toFixed(1)} km`],
      ["Speed",      `${s.speed?.toFixed(3)} km/s`],
      ["ECI X",      `${s.x?.toFixed(0)} km`],
      ["ECI Y",      `${s.y?.toFixed(0)} km`],
      ["ECI Z",      `${s.z?.toFixed(0)} km`],
      ["Updated",    `${(s.timestamp || "").substring(11, 19)} UTC`],
    ];

    body.innerHTML = rows.map(([k, v]) => `
      <div class="d-row">
        <span class="d-key">${k}</span>
        <span class="d-val c">${esc(String(v ?? "—"))}</span>
      </div>`).join("");
  },

  renderRiskCards(summary) {
    ["crit","high","med","low"].forEach(k => {
      const el = document.getElementById(`rc-${k}`);
      if (el) el.textContent = summary[{ crit:"critical", high:"high", med:"medium", low:"low" }[k]] ?? "0";
    });
  },

  renderEvents(events) {
    const el = document.getElementById("ev-list");
    if (!el) return;
    el.innerHTML = "";
    if (!events.length) { el.innerHTML = '<div class="empty">No events detected</div>'; return; }

    events.slice(0, 12).forEach((e, i) => {
      const lvl  = e.risk_level.toLowerCase().replace("medium","med");
      const card = document.createElement("div");
      card.className = `ev-card ${lvl}`;
      card.style.animationDelay = `${i * 0.05}s`;
      card.innerHTML = `
        <div class="ev-top">
          <span class="ev-badge">${e.risk_level}</span>
          <span class="ev-km">${e.distance_km.toFixed(2)} km</span>
        </div>
        <div class="ev-sats">${esc(e.sat1_name)} ↔ ${esc(e.sat2_name)}</div>
      `;
      el.appendChild(card);
    });
  },

  renderAnomalies(anoms) {
    const el = document.getElementById("an-list");
    if (!el) return;
    el.innerHTML = "";
    if (!anoms.length) { el.innerHTML = '<div class="empty">No anomalies detected</div>'; return; }

    anoms.slice(0, 8).forEach((a, i) => {
      const card = document.createElement("div");
      card.className = "an-card";
      card.style.animationDelay = `${i * 0.05}s`;
      card.innerHTML = `
        <div class="an-name">${esc(a.name)}</div>
        <div class="an-meta">Score: ${a.anomaly_score.toFixed(4)} · NORAD ${a.norad_id}</div>
        <div class="an-reason">${esc(a.reason)}</div>
      `;
      el.appendChild(card);
    });
  },

  showTooltip(s, x, y) {
    const tip = document.getElementById("tooltip");
    if (!tip || !s?.norad_id) return;
    document.getElementById("tt-name").textContent = s.name;
    document.getElementById("tt-body").innerHTML = [
      ["Alt",   `${s.altitude?.toFixed(1)} km`],
      ["Speed", `${s.speed?.toFixed(2)} km/s`],
      ["Lat",   `${s.latitude?.toFixed(2)}°`],
      ["Lon",   `${s.longitude?.toFixed(2)}°`],
    ].map(([k, v]) => `
      <div class="d-row">
        <span class="d-key">${k}</span>
        <span class="d-val c">${v}</span>
      </div>`).join("");
    tip.style.left = `${x + 16}px`;
    tip.style.top  = `${y - 10}px`;
    tip.classList.add("show");
  },

  hideTooltip() { document.getElementById("tooltip")?.classList.remove("show"); },

  spawnRadarBlips() {
    const disc = document.querySelector(".radar-disc");
    if (!disc) return;
    const colors = ["#00d4ff","#00ffcc","#ffaa00","#ff2255"];
    [
      { top:"26%", left:"62%" },
      { top:"60%", left:"34%" },
      { top:"72%", left:"68%" },
      { top:"38%", left:"50%" },
      { top:"48%", left:"22%" },
    ].forEach((pos, i) => {
      const b = document.createElement("div");
      b.className = "r-blip";
      b.style.cssText = `top:${pos.top};left:${pos.left};background:${colors[i%colors.length]};box-shadow:0 0 8px ${colors[i%colors.length]};animation-delay:${i*0.7}s;`;
      disc.appendChild(b);
    });
  },
};

/* ═══════════════════════════════════════════════════════ APP */
class App {
  constructor() {
    this.api   = new APIClient(CFG.API_BASE);
    this.globe = new GlobeRenderer(document.getElementById("globe-canvas"));
    this._sats = [];
    this._collEvts = [];
    this._group    = CFG.DEFAULT_GROUP;
    this._selSat   = null;
    this._showOrbits = true;
    this._showConj   = true;
    this._showTrails = true;
    this._showLabels = false;
  }

  async start() {
    UI._startClock();
    UI.spawnRadarBlips();
    this.globe.init();

    /* Wire globe callbacks */
    this.globe.onHoverSat = (s, x, y) => s ? UI.showTooltip(s, x, y) : UI.hideTooltip();
    this.globe.onClickSat = s => this._selectSat(s);

    /* Wire controls */
    this._wireControls();

    /* Health check */
    try {
      await this.api.health();
      UI.setMetric("mc-api", "OK", "green");
    } catch {
      UI.setMetric("mc-api", "ERR", "red");
    }

    /* Initial data load */
    await this._loadSats();
    setTimeout(() => this._loadCollisions(), 1500);
    setTimeout(() => this._loadAnomalies(),  4200);

    /* Polling */
    setInterval(() => this._refreshSatPositions(), CFG.POLL_SATS_MS);
    setInterval(() => this._loadCollisions(),       CFG.POLL_COLL_MS);
    setInterval(() => this._loadAnomalies(),        CFG.POLL_ANOM_MS);
  }

  /* ── Data loaders ─────────────────────────────────────── */
  async _loadSats() {
    const lo = document.getElementById("sat-loading");
    if (lo) lo.style.display = "block";
    try {
      const r = await this.api.satellites(this._group, CFG.LIMIT);
      this._sats = r.data;
      this.globe._currentGroup = this._group;
      this.globe.addSatellites(this._sats, this._group);
      UI.renderSatList(this._sats, s => this._selectSat(s));
      UI.setMetric("mc-sats", this._sats.length, "cyan");
      UI.setBadge("badge-sats", this._sats.length, "cyan");
      const hc = document.getElementById("hud-count");
      if (hc) hc.textContent = this._sats.length;
    } catch (e) {
      console.warn("Satellite load failed:", e.message);
    } finally {
      if (lo) lo.style.display = "none";
    }
  }

  async _refreshSatPositions() {
    try {
      const r = await this.api.satellites(this._group, CFG.LIMIT);
      this._sats = r.data;
      this.globe.updateSatPositions(this._sats);
    } catch { /* silent */ }
  }

  async _loadCollisions() {
    try {
      const r   = await this.api.collisions(this._group, CFG.LIMIT);
      const sum = r.data.summary || {};
      this._collEvts = r.data.events || [];

      UI.renderRiskCards(sum);
      UI.renderEvents(this._collEvts);
      UI.setMetric("mc-coll", sum.total_events ?? 0, (sum.critical ?? 0) > 0 ? "red" : "amber");
      UI.setBadge("badge-coll", sum.total_events ?? 0, (sum.critical ?? 0) > 0 ? "red" : "amber");

      if (this._showConj && this._collEvts.length) {
        this.globe.showConjArcs(this._collEvts, this._sats);
      }
    } catch (e) { console.warn("Collision load:", e.message); }
  }

  async _loadAnomalies() {
    try {
      const r    = await this.api.anomalies(this._group, CFG.LIMIT);
      const anoms = r.data.anomalies || [];
      UI.renderAnomalies(anoms);
      UI.setMetric("mc-anom", anoms.length, anoms.length > 0 ? "amber" : "green");
      UI.setBadge("badge-anom", anoms.length, anoms.length > 0 ? "amber" : "cyan");

      /* Highlight anomalous satellites in the globe */
      this.globe.flagAnomalousSats(anoms.map(a => a.norad_id));
    } catch (e) { console.warn("Anomaly load:", e.message); }
  }

  /* ── Satellite selection ──────────────────────────────── */
  async _selectSat(sat) {
    if (!sat?.norad_id) return;
    this._selSat = sat;
    UI.showSatDetail(sat);

    if (this._showOrbits) {
      try {
        const r = await this.api.orbit(sat.norad_id, this._group, CFG.ORBIT_STEPS);
        this.globe.addOrbitTrack(r.data.track, this._group);
      } catch { /* no orbit data */ }
    }
  }

  /* ── Controls wiring ──────────────────────────────────── */
  _wireControls() {
    /* Group selector */
    document.getElementById("grp-sel")?.addEventListener("change", async e => {
      this._group = e.target.value;
      this.globe.clearOrbitLines();
      this.globe.clearConjLines();
      await this._loadSats();
      this._loadCollisions();
    });

    /* Orbit toggle */
    this._wireToggle("btn-orbits", val => {
      this._showOrbits = val;
      this.globe.setShowOrbits(val);
    });

    /* Conjunction toggle */
    this._wireToggle("btn-conj", val => {
      this._showConj = val;
      this.globe.setShowConj(val);
      if (val && this._collEvts.length) this.globe.showConjArcs(this._collEvts, this._sats);
    });

    /* Trail toggle */
    this._wireToggle("btn-trails", val => {
      this._showTrails = val;
      this.globe.setShowTrails(val);
    });

    /* Labels toggle (cosmetic for now) */
    this._wireToggle("btn-labels", val => { this._showLabels = val; });

    /* Refresh buttons */
    document.getElementById("btn-refresh")?.addEventListener("click", () => {
      this._loadSats();
      this._loadCollisions();
      this._loadAnomalies();
    });

    document.getElementById("btn-refresh-coll")?.addEventListener("click", () => this._loadCollisions());
    document.getElementById("btn-refresh-anom")?.addEventListener("click", () => this._loadAnomalies());

    /* Retrain */
    document.getElementById("btn-retrain")?.addEventListener("click", async e => {
      const orig = e.target.textContent;
      e.target.textContent = "⟳ Training...";
      try { await this._loadAnomalies(); } finally { e.target.textContent = orig; }
    });

    /* Orbit track from detail panel */
    document.getElementById("btn-orbit-track")?.addEventListener("click", () => {
      if (this._selSat) this._selectSat(this._selSat);
    });
  }

  _wireToggle(id, cb) {
    const btn = document.getElementById(id);
    if (!btn) return;
    let state = true;
    btn.addEventListener("click", () => {
      state = !state;
      btn.classList.toggle("on", state);
      cb(state);
    });
  }
}

/* ═══════════════════════════════════════════════════ HELPERS */
function esc(s) {
  return String(s)
    .replace(/&/g,"&amp;").replace(/</g,"&lt;")
    .replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

/* ═══════════════════════════════════════════════════════ BOOT */
window.addEventListener("DOMContentLoaded", () => {
  const app = new App();
  app.start().catch(console.error);
  window._app = app;   // expose for devtools debugging
});