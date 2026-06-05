// ── PartField Three.js Viewer ─────────────────────────────────────────────────
(function () {
  'use strict';

  // MUST match backend PALETTE and app.js PALETTE exactly
  const PALETTE = [
    "#E63946","#2A9D8F","#E9C46A","#457B9D","#8338EC",
    "#06D6A0","#FF6B6B","#118AB2","#F72585","#3A86FF",
    "#FB5607","#8AC926","#FFD166","#4CC9F0","#7B2D8B",
    "#EF476F","#00B4D8","#90123F","#50C878","#FF8C00",
    "#6432C8","#149650","#C85014","#3264C8",
  ];

  let renderer, scene, camera, controls;
  let rootGroup     = null;
  let meshGroups    = [];
  let labelSprites  = [];
  let currentLabels = [];
  let wireframeOn   = false;
  let labelsVisible = true;
  let viewMode      = 'full';
  let explodeAmount = 0;

  const canvas     = document.getElementById('threeCanvas');
  const canvasWrap = document.getElementById('canvasWrap');

  function init() {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x060a0f, 1);
    renderer.shadowMap.enabled = true;
    resize();

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x060a0f);
    camera = new THREE.PerspectiveCamera(42, 1, 0.001, 2000);
    camera.position.set(0, 2, 5);

    scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const sun = new THREE.DirectionalLight(0xffffff, 1.0); sun.position.set(4,8,6); scene.add(sun);
    const fill = new THREE.DirectionalLight(0x4488ff, 0.4); fill.position.set(-6,2,-4); scene.add(fill);
    const rim = new THREE.DirectionalLight(0xff6600, 0.25); rim.position.set(0,-4,8); scene.add(rim);

    const grid = new THREE.GridHelper(12, 24, 0x1c2d3e, 0x0f1824);
    grid.material.transparent = true; grid.material.opacity = 0.5; scene.add(grid);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true; controls.dampingFactor = 0.07;
    controls.minDistance = 0.1; controls.maxDistance = 200;

    window.addEventListener('resize', resize);
    animate();
  }

  function resize() {
    const w = canvasWrap.offsetWidth || 800, h = canvasWrap.offsetHeight || 600;
    renderer.setSize(w, h, false);
    if (camera) { camera.aspect = w/h; camera.updateProjectionMatrix(); }
  }

  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
    if (labelsVisible) _updateLabels();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Load original model (blob URL preview before segmentation)
  // ─────────────────────────────────────────────────────────────────────────
  window.loadModel = function(url, fileExt, labels) {
    _clearScene();
    const ext = (fileExt||'glb').toLowerCase();
    function onLoaded(obj) {
      let root = (obj && obj.scene) ? obj.scene : obj;
      if (root instanceof THREE.BufferGeometry) {
        root = new THREE.Mesh(root, new THREE.MeshStandardMaterial({
          vertexColors: !!root.attributes.color, color: 0x888888, roughness:0.55, metalness:0.1
        }));
      }
      _addRoot(root, labels, false);
    }
    const onErr = e => console.error('Loader error:', e);
    if (ext==='glb'||ext==='gltf') new THREE.GLTFLoader().load(url, onLoaded, undefined, onErr);
    else if (ext==='obj') new THREE.OBJLoader().load(url, onLoaded, undefined, onErr);
    else if (ext==='ply') new THREE.PLYLoader().load(url, geo=>onLoaded(geo), undefined, onErr);
    else if (ext==='stl') new THREE.STLLoader().load(url, geo=>onLoaded(geo), undefined, onErr);
    else new THREE.GLTFLoader().load(url, onLoaded, undefined,
           ()=>new THREE.OBJLoader().load(url, onLoaded, undefined, onErr));
  };

  // ─────────────────────────────────────────────────────────────────────────
  // Load colored PLY (post-segmentation) — splits into per-cluster meshes
  // Falls back to original model URL if PLY fails
  // ─────────────────────────────────────────────────────────────────────────
  window.loadModelWithFallback = function(primaryUrl, primaryExt, fallbackUrl, fallbackExt, labels, FL) {
    _clearScene();
    function tryFallback() {
      console.log('Colored PLY failed — falling back to original model');
      _clearScene();
      window.loadModel(fallbackUrl, fallbackExt, labels);
    }
    if ((primaryExt||'').toLowerCase() === 'ply') {
      new THREE.PLYLoader().load(
        primaryUrl,
        geo => {
          if (!geo || !geo.attributes.position || geo.attributes.position.count === 0) {
            console.error('PLY geometry empty — falling back');
            return tryFallback();
          }
          const nV = geo.attributes.position.count;
          const nF = geo.index ? geo.index.count / 3 : Math.floor(nV / 3);
          console.log(`PLY loaded: nV=${nV}, indexed=${!!geo.index}, nF=${nF}, FL.length=${FL ? FL.length : 0}, labels=${labels ? labels.length : 0}`);

          if (FL && FL.length > 0) {
            const flMatch = Math.abs(FL.length - nF) <= Math.max(10, nF * 0.01);
            if (flMatch) {
              // For large models (>20 parts), use vertex-color single mesh for speed
              // For small models, split into individual meshes for explode/isolate
              if (labels.length > 20) {
                console.log(`${labels.length} parts — using fast vertex-color mode`);
                _buildVertexColoredMesh(geo, labels, FL);
              } else {
                console.log(`${labels.length} parts — using split-mesh mode`);
                _buildSplitMeshes(geo, labels, FL);
              }
            } else {
              console.warn(`FL/nF mismatch: FL=${FL.length} nF=${nF} — vertex-color fallback`);
              if (geo.attributes.color) _buildSplitMeshesFromColors(geo, labels);
              else _buildVertexColoredMesh(geo, labels, FL);
            }
          } else if (labels && labels.length > 0 && geo.attributes.color) {
            _buildSplitMeshesFromColors(geo, labels);
          } else {
            const mat = new THREE.MeshStandardMaterial({
              vertexColors: !!geo.attributes.color, color: 0x888888, roughness:0.55, metalness:0.1
            });
            _addRoot(new THREE.Mesh(geo, mat), labels, true);
          }
        },
        undefined,
        tryFallback
      );
    } else {
      window.loadModel(primaryUrl, primaryExt, labels);
    }
  };

  // ─────────────────────────────────────────────────────────────────────────
  // Fast vertex-colored single mesh — for large models with many parts
  // Colors each vertex by its cluster's palette color using FL array
  // Supports highlight/hover but not explode/isolate (too many parts)
  // ─────────────────────────────────────────────────────────────────────────
  function _buildVertexColoredMesh(geo, labels, FL) {
    const posAttr = geo.attributes.position;
    const nV = posAttr.count;
    const nF = geo.index ? geo.index.count / 3 : Math.floor(nV / 3);
    const nFeff = Math.min(FL.length, nF);

    // Build cluster_id → RGB color map
    const cidToColor = {};
    labels.forEach(lbl => {
      const c = new THREE.Color(lbl.color);
      cidToColor[lbl.cluster_id] = [c.r, c.g, c.b];
    });

    // Per-vertex color: vote from faces containing each vertex
    const vertColors = new Float32Array(nV * 3).fill(0.5);
    const vertVotes  = new Int32Array(nV).fill(-1);

    for (let fi = 0; fi < nFeff; fi++) {
      const cid = FL[fi];
      const rgb = cidToColor[cid] || [0.5, 0.5, 0.5];
      for (let k = 0; k < 3; k++) {
        const vi = geo.index ? geo.index.getX(fi*3+k) : fi*3+k;
        if (vi < nV) {
          vertColors[vi*3]   = rgb[0];
          vertColors[vi*3+1] = rgb[1];
          vertColors[vi*3+2] = rgb[2];
          vertVotes[vi] = cid;
        }
      }
    }

    geo.setAttribute('color', new THREE.BufferAttribute(vertColors, 3));
    if (!geo.attributes.normal) geo.computeVertexNormals();

    const mat  = new THREE.MeshStandardMaterial({
      vertexColors: true, roughness: 0.5, metalness: 0.05, side: THREE.DoubleSide
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.castShadow = true;

    // One group per label for hover highlighting
    meshGroups = labels.map((lbl, gi) => ({
      meshes: [mesh], clusterId: lbl.cluster_id, groupIdx: gi,
      color: lbl.color,
      centroid: new THREE.Vector3(...(lbl.centroid || [0,0,0])),
      origPositions: [new THREE.Vector3()],
    }));

    const root = new THREE.Group();
    root.add(mesh);

    // Normalise
    root.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(root);
    if (box.isEmpty()) { console.error('Empty box in vertexColored'); return; }
    const bc = new THREE.Vector3(); box.getCenter(bc);
    const bs = new THREE.Vector3(); box.getSize(bs);
    const sc = 3 / Math.max(bs.x, bs.y, bs.z, 0.001);
    root.position.set(-bc.x*sc, -bc.y*sc, -bc.z*sc);
    root.scale.setScalar(sc);
    meshGroups.forEach(g => {
      g.centroid.set(g.centroid.x*sc-bc.x*sc, g.centroid.y*sc-bc.y*sc, g.centroid.z*sc-bc.z*sc);
    });

    scene.add(root); rootGroup = root;
    _fitCamera(root);
    currentLabels = labels;
    _buildLabelSprites(labels);
    console.log(`Built vertex-colored mesh: ${labels.length} parts`);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Build split meshes using FL (face→cluster index) directly from backend
  // This is the correct approach — no color reverse-engineering needed
  // ─────────────────────────────────────────────────────────────────────────
  function _buildSplitMeshes(geo, labels, FL) {
    const posAttr  = geo.attributes.position;
    const normAttr = geo.attributes.normal;
    const nV = posAttr.count;

    // Our backend writes a NON-INDEXED PLY: nV = nF*3
    // so face fi uses vertices [fi*3, fi*3+1, fi*3+2]
    // FL.length == nF == nV/3  → perfect 1:1 mapping
    const nF = geo.index ? geo.index.count / 3 : Math.floor(nV / 3);

    console.log(`PLY: nV=${nV}, nF=${nF}, FL.length=${FL ? FL.length : 0}, labels=${labels.length}`);

    if (!FL || FL.length === 0) {
      console.warn('No FL — fallback to vertex-color split');
      _buildSplitMeshesFromColors(geo, labels);
      return;
    }

    if (Math.abs(FL.length - nF) > 10) {
      console.warn(`FL/nF mismatch: FL=${FL.length} nF=${nF}. Attempting remap.`);
    }

    // Build per-cluster vertex position arrays directly
    // For non-indexed PLY: face fi → verts fi*3, fi*3+1, fi*3+2
    const nLabels = labels.length;
    const cidToIdx = {};
    labels.forEach((l, i) => { cidToIdx[l.cluster_id] = i; });

    // Count verts per cluster first for pre-allocation
    const clusterVertCount = new Array(nLabels).fill(0);
    const nFeff = Math.min(FL.length, nF);
    for (let fi = 0; fi < nFeff; fi++) {
      const cid = FL[fi];
      const idx = cidToIdx[cid];
      if (idx !== undefined) clusterVertCount[idx] += 3;
    }

    // Allocate buffers
    const posBuffers  = labels.map((_, i) => new Float32Array(clusterVertCount[i] * 3));
    const normBuffers = normAttr
      ? labels.map((_, i) => new Float32Array(clusterVertCount[i] * 3))
      : null;
    const offsets = new Array(nLabels).fill(0);

    // Fill buffers — each face writes 3 vertices into the cluster buffer
    // offsets[gi] = next write position in posBuffers[gi] (units = vertices, not floats)
    for (let fi = 0; fi < nFeff; fi++) {
      const cid = FL[fi];
      const gi  = cidToIdx[cid];
      if (gi === undefined) continue;

      for (let k = 0; k < 3; k++) {
        // Vertex index in the source PLY geometry
        const vi = geo.index
          ? geo.index.getX(fi * 3 + k)
          : fi * 3 + k;                  // non-indexed: sequential

        // Write position into cluster buffer at (offsets[gi] + k) * 3
        const dst = (offsets[gi] + k) * 3;
        posBuffers[gi][dst]     = posAttr.getX(vi);
        posBuffers[gi][dst + 1] = posAttr.getY(vi);
        posBuffers[gi][dst + 2] = posAttr.getZ(vi);
        if (normBuffers) {
          normBuffers[gi][dst]     = normAttr.getX(vi);
          normBuffers[gi][dst + 1] = normAttr.getY(vi);
          normBuffers[gi][dst + 2] = normAttr.getZ(vi);
        }
      }
      offsets[gi] += 3;  // advance by 3 verts (one face)
    }

    const root = new THREE.Group();
    meshGroups = [];

    labels.forEach((lbl, gi) => {
      const pos = posBuffers[gi];
      if (!pos || pos.length === 0 || clusterVertCount[gi] === 0) {
        console.warn(`Cluster ${lbl.cluster_id} (${lbl.label}) has 0 verts — skipping`);
        return;
      }

      const subGeo = new THREE.BufferGeometry();
      subGeo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      if (normBuffers && normBuffers[gi].length > 0) {
        subGeo.setAttribute('normal', new THREE.BufferAttribute(normBuffers[gi], 3));
      } else {
        subGeo.computeVertexNormals();
      }

      const col3 = new THREE.Color(lbl.color);
      const mat  = new THREE.MeshStandardMaterial({
        color: col3, emissive: col3, emissiveIntensity: 0.10,
        roughness: 0.45, metalness: 0.05,
        side: THREE.DoubleSide,
      });
      const mesh = new THREE.Mesh(subGeo, mat);
      mesh.castShadow = true; mesh.receiveShadow = true;
      root.add(mesh);

      // Centroid from position buffer
      const n = pos.length / 3;
      const cen = new THREE.Vector3();
      for (let i = 0; i < n; i++) cen.x += pos[i*3], cen.y += pos[i*3+1], cen.z += pos[i*3+2];
      cen.divideScalar(n);

      meshGroups.push({
        meshes: [mesh], clusterId: lbl.cluster_id, groupIdx: gi,
        color: lbl.color, centroid: cen, origPositions: [new THREE.Vector3()],
      });

      console.log(`  ${lbl.label}: ${n} verts, color ${lbl.color}`);
    });

    if (meshGroups.length === 0) {
      console.error('No meshes built — all clusters empty. Falling back.');
      _buildSplitMeshesFromColors(geo, labels);
      return;
    }

    // Normalise to 3-unit cube
    const box = new THREE.Box3().setFromObject(root);
    if (box.isEmpty()) { console.error('Empty box'); return; }
    const bc  = new THREE.Vector3(); box.getCenter(bc);
    const bs  = new THREE.Vector3(); box.getSize(bs);
    const sc  = 3 / Math.max(bs.x, bs.y, bs.z, 0.001);
    root.position.set(-bc.x*sc, -bc.y*sc, -bc.z*sc);
    root.scale.setScalar(sc);

    meshGroups.forEach(g => {
      g.centroid.set(g.centroid.x*sc - bc.x*sc,
                     g.centroid.y*sc - bc.y*sc,
                     g.centroid.z*sc - bc.z*sc);
    });

    scene.add(root); rootGroup = root;
    _fitCamera(root);
    currentLabels = labels;
    _buildLabelSprites(labels);
    console.log(`Built ${meshGroups.length} colored mesh groups.`);
  }

  // Fallback: split by vertex color matching (used when FL not available)
  function _buildSplitMeshesFromColors(geo, labels) {
    const posAttr = geo.attributes.position;
    const colAttr = geo.attributes.color;
    const normAttr = geo.attributes.normal;
    const nV = posAttr.count;

    const colorToCid = {};
    labels.forEach(lbl => {
      const c = new THREE.Color(lbl.color);
      const key = `${Math.round(c.r*255)},${Math.round(c.g*255)},${Math.round(c.b*255)}`;
      colorToCid[key] = lbl.cluster_id;
    });

    const vertCid = new Int32Array(nV).fill(labels[0]?.cluster_id || 0);
    if (colAttr) {
      for (let v = 0; v < nV; v++) {
        const r = Math.round(colAttr.getX(v)*255);
        const g = Math.round(colAttr.getY(v)*255);
        const b = Math.round(colAttr.getZ(v)*255);
        const key = `${r},${g},${b}`;
        if (colorToCid[key] !== undefined) {
          vertCid[v] = colorToCid[key];
        } else {
          let bestCid = labels[0].cluster_id, bestD = Infinity;
          for (const [ck, cid] of Object.entries(colorToCid)) {
            const [cr,cg,cb] = ck.split(',').map(Number);
            const d = (r-cr)**2 + (g-cg)**2 + (b-cb)**2;
            if (d < bestD) { bestD = d; bestCid = cid; }
          }
          vertCid[v] = bestCid;
        }
      }
    }

    const nF = geo.index ? geo.index.count/3 : nV/3;
    const cidToFaceVerts = {};
    labels.forEach(l => { cidToFaceVerts[l.cluster_id] = []; });
    for (let fi = 0; fi < nF; fi++) {
      let v0,v1,v2;
      if (geo.index) { v0=geo.index.getX(fi*3); v1=geo.index.getX(fi*3+1); v2=geo.index.getX(fi*3+2); }
      else { v0=fi*3; v1=fi*3+1; v2=fi*3+2; }
      const votes = {};
      [v0,v1,v2].forEach(v => { const c=vertCid[v]; votes[c]=(votes[c]||0)+1; });
      const cid = parseInt(Object.entries(votes).sort((a,b)=>b[1]-a[1])[0][0]);
      if (cidToFaceVerts[cid]) cidToFaceVerts[cid].push(v0,v1,v2);
    }

    const root = new THREE.Group();
    meshGroups = [];
    labels.forEach((lbl, gi) => {
      const verts = cidToFaceVerts[lbl.cluster_id];
      if (!verts || !verts.length) return;
      const posArr = new Float32Array(verts.length*3);
      verts.forEach((vi,i) => { posArr[i*3]=posAttr.getX(vi); posArr[i*3+1]=posAttr.getY(vi); posArr[i*3+2]=posAttr.getZ(vi); });
      const subGeo = new THREE.BufferGeometry();
      subGeo.setAttribute('position', new THREE.BufferAttribute(posArr,3));
      subGeo.computeVertexNormals();
      const col3 = new THREE.Color(lbl.color);
      const mesh = new THREE.Mesh(subGeo, new THREE.MeshStandardMaterial({color:col3,emissive:col3,emissiveIntensity:0.08,roughness:0.5,metalness:0.1}));
      root.add(mesh);
      const cen = new THREE.Vector3();
      for (let i=0;i<verts.length;i++) cen.x+=posArr[i*3],cen.y+=posArr[i*3+1],cen.z+=posArr[i*3+2];
      cen.divideScalar(verts.length);
      meshGroups.push({meshes:[mesh],clusterId:lbl.cluster_id,groupIdx:gi,color:lbl.color,centroid:cen,origPositions:[new THREE.Vector3()]});
    });
    const box=new THREE.Box3().setFromObject(root);
    const cen=new THREE.Vector3(); box.getCenter(cen);
    const siz=new THREE.Vector3(); box.getSize(siz);
    const sc=3/Math.max(siz.x,siz.y,siz.z,0.001);
    root.position.set(-cen.x*sc,-cen.y*sc,-cen.z*sc); root.scale.setScalar(sc);
    meshGroups.forEach(g=>{g.centroid.set(g.centroid.x*sc-cen.x*sc,g.centroid.y*sc-cen.y*sc,g.centroid.z*sc-cen.z*sc);});
    scene.add(root); rootGroup=root; _fitCamera(root); currentLabels=labels; _buildLabelSprites(labels);
  }


  // ─────────────────────────────────────────────────────────────────────────
  // Add a root object (GLB/OBJ path)
  // ─────────────────────────────────────────────────────────────────────────
  function _addRoot(root, labels, skipColor) {
    const box = new THREE.Box3().setFromObject(root);
    const cen  = new THREE.Vector3(); box.getCenter(cen);
    const siz  = new THREE.Vector3(); box.getSize(siz);
    const sc   = 3 / Math.max(siz.x, siz.y, siz.z, 0.001);
    // Correct: set position explicitly so cen is not mutated before use
    root.position.set(-cen.x * sc, -cen.y * sc, -cen.z * sc);
    root.scale.setScalar(sc);
    scene.add(root); rootGroup = root;
    if (!skipColor && labels && labels.length) _assignColors(root, labels);
    else _collectGroups(root, labels);
    _fitCamera(root);
    if (labels && labels.length) { currentLabels = labels; _buildLabelSprites(labels); }
  }

  function _assignColors(root, labels) {
    meshGroups = [];
    const meshes = [];
    root.traverse(c => { if (c.isMesh) meshes.push(c); });
    if (!meshes.length) return;
    const n = labels.length, pg = Math.max(1, Math.ceil(meshes.length/n));
    meshes.forEach((mesh, i) => {
      const gi = Math.min(Math.floor(i/pg), n-1);
      const lbl = labels[gi];
      const col = new THREE.Color(lbl ? lbl.color : '#888888');
      mesh.material = new THREE.MeshStandardMaterial({ color:col, emissive:col, emissiveIntensity:0.06, roughness:0.55, metalness:0.1 });
      mesh.castShadow = true; mesh.receiveShadow = true;
      const wb = new THREE.Box3().setFromObject(mesh);
      const wcen = new THREE.Vector3(); wb.getCenter(wcen);
      let grp = meshGroups.find(g => g.groupIdx===gi);
      if (!grp) { grp={ meshes:[], clusterId: lbl?lbl.cluster_id:i, groupIdx:gi, color:lbl?lbl.color:'#888', centroid:new THREE.Vector3(), origPositions:[] }; meshGroups.push(grp); }
      grp.meshes.push(mesh); grp.origPositions.push(mesh.position.clone()); grp.centroid.add(wcen);
    });
    meshGroups.forEach(g => { if(g.meshes.length>1) g.centroid.divideScalar(g.meshes.length); });
  }

  function _collectGroups(root, labels) {
    meshGroups = [];
    const meshes = [];
    root.traverse(c => { if (c.isMesh) meshes.push(c); });
    if (!meshes.length) return;
    const n = labels ? labels.length : 1, pg = Math.max(1, Math.ceil(meshes.length/n));
    meshes.forEach((mesh, i) => {
      const gi = Math.min(Math.floor(i/pg), n-1);
      const lbl = labels ? labels[gi] : null;
      const wb = new THREE.Box3().setFromObject(mesh); const wcen = new THREE.Vector3(); wb.getCenter(wcen);
      let grp = meshGroups.find(g=>g.groupIdx===gi);
      if (!grp) { grp={ meshes:[], clusterId:lbl?lbl.cluster_id:i, groupIdx:gi, color:lbl?lbl.color:'#888', centroid:new THREE.Vector3(), origPositions:[] }; meshGroups.push(grp); }
      grp.meshes.push(mesh); grp.origPositions.push(mesh.position.clone()); grp.centroid.add(wcen);
    });
    meshGroups.forEach(g => { if(g.meshes.length>1) g.centroid.divideScalar(g.meshes.length); });
  }

  window.applyLabels = function(labels) {
    currentLabels = labels||[];
    meshGroups.forEach(grp => {
      const lbl = labels.find(l=>l.cluster_id===grp.clusterId)||labels[grp.groupIdx];
      if (!lbl) return;
      grp.color=lbl.color; grp.clusterId=lbl.cluster_id;
      const col = new THREE.Color(lbl.color);
      grp.meshes.forEach(m => { if(m.material&&!m.geometry.attributes.color){ m.material.color.set(col); m.material.emissive.set(col); m.material.emissiveIntensity=0.06; }});
    });
    _buildLabelSprites(labels);
  };

  function _buildLabelSprites(labels) {
    const layer = document.getElementById('labelLayer');
    layer.innerHTML=''; labelSprites=[];
    if (!labels||!labels.length||!meshGroups.length) return;
    meshGroups.forEach(grp => {
      if (!grp.meshes.length) return;
      const lbl = labels.find(l=>l.cluster_id===grp.clusterId)||labels[grp.groupIdx];
      if (!lbl) return;
      const el = document.createElement('div');
      el.className='part-label'; el.textContent=lbl.label.replace(/_/g,' ');
      el.style.borderColor=lbl.color; el.style.color=lbl.color;
      layer.appendChild(el);
      labelSprites.push({el, mesh:grp.meshes[0]});
    });
  }

  function _updateLabels() {
    if (!labelSprites.length||!camera||!canvasWrap) return;
    const w=canvasWrap.offsetWidth, h=canvasWrap.offsetHeight;
    labelSprites.forEach(({el, mesh}) => {
      if (!mesh.visible) { el.style.opacity='0'; return; }
      const box=new THREE.Box3().setFromObject(mesh);
      const top=new THREE.Vector3(); box.getCenter(top); top.y=box.max.y;
      const proj=top.clone().project(camera);
      const x=(proj.x*0.5+0.5)*w, y=(proj.y*-0.5+0.5)*h;
      el.style.opacity = (proj.z>1||x<-50||x>w+50||y<-20||y>h+20) ? '0':'1';
      el.style.left=x+'px'; el.style.top=y+'px';
    });
  }

  // ─────────────────────────────────────────────────────────────────────────
  // View modes: Full / Explode / Isolate
  // ─────────────────────────────────────────────────────────────────────────
  window.setViewMode = function(mode) {
    viewMode = mode;
    document.querySelectorAll('.tb-btn').forEach(b => b.classList.remove('active'));
    const btn = document.getElementById('btn-' + mode);
    if (btn) btn.classList.add('active');

    const ec = document.getElementById('explodeCtrl');
    if (mode === 'full') {
      if (ec) ec.style.display = 'none';
      _resetExplode();
      _showAll();
    } else if (mode === 'explode') {
      if (ec) ec.style.display = 'flex';
      const sl = document.getElementById('explodeSlider');
      const vl = document.getElementById('explodeVal');
      if (sl) sl.value = 0;
      if (vl) vl.textContent = '0%';
      _showAll();
      _resetExplode();
    } else if (mode === 'isolate') {
      if (ec) ec.style.display = 'none';
      _resetExplode();
      openIsolateMode();
    }
  };

  window.setExplode = function(t) {
    explodeAmount = parseFloat(t);
    const vl = document.getElementById('explodeVal');
    if (vl) vl.textContent = Math.round(t * 100) + '%';
    if (!meshGroups.length) return;

    // Compute true model centre from bounding box of all meshes
    const allBox = new THREE.Box3();
    meshGroups.forEach(g => g.meshes.forEach(m => {
      m.position.copy(g.origPositions[g.meshes.indexOf(m)]);  // reset first
      allBox.expandByObject(m);
    }));
    const mc = new THREE.Vector3(); allBox.getCenter(mc);

    meshGroups.forEach(grp => {
      // Explode direction: from model centre → group centroid
      const dir = grp.centroid.clone().sub(mc);
      const len = dir.length();
      const dn  = len > 0.001 ? dir.normalize() : new THREE.Vector3(0, 1, 0);

      // Scale explosion by group size so small parts don't disappear
      const scale = Math.max(0.6, 1.0 + grp.centroid.distanceTo(mc) * 0.3);
      const mag   = 3.5 * explodeAmount * scale;

      grp.meshes.forEach((mesh, mi) => {
        mesh.position.copy(grp.origPositions[mi]).addScaledVector(dn, mag);
      });
    });
  };

  function _resetExplode() {
    meshGroups.forEach(g => g.meshes.forEach((m, i) => m.position.copy(g.origPositions[i])));
    const sl = document.getElementById('explodeSlider');
    const vl = document.getElementById('explodeVal');
    if (sl) sl.value = 0;
    if (vl) vl.textContent = '0%';
    explodeAmount = 0;
  }

  function _showAll() {
    meshGroups.forEach(grp => grp.meshes.forEach(mesh => {
      mesh.visible = true;
      if (mesh.material) {
        mesh.material.opacity     = 1;
        mesh.material.transparent = false;
        mesh.material.depthWrite  = true;
        if (!mesh.geometry.attributes.color) mesh.material.emissiveIntensity = 0.06;
      }
    }));
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Isolate — click a part to focus it, others fade to ghost
  // ─────────────────────────────────────────────────────────────────────────
  window.openIsolateMode = function() {
    const m = document.getElementById('isolateModal');
    if (m) { m.style.display = 'flex'; _buildIsolateGrid(); }
  };
  window.closeIsolate = function() {
    const m = document.getElementById('isolateModal');
    if (m) m.style.display = 'none';
  };

  function _buildIsolateGrid() {
    const grid = document.getElementById('imGrid');
    grid.innerHTML = '';

    // Show All card
    const all = document.createElement('div');
    all.className = 'im-card';
    all.innerHTML = `
      <div class="im-thumb" style="background:#1c2d3e;font-size:26px;color:#22d97b;
           display:flex;align-items:center;justify-content:center;">⬡</div>
      <div class="im-label" style="color:#22d97b">Show All</div>`;
    all.onclick = () => { _showAll(); _resetExplode(); closeIsolate(); setViewMode('full'); };
    grid.appendChild(all);

    meshGroups.forEach(grp => {
      const lbl = currentLabels.find(l => l.cluster_id === grp.clusterId)
                  || currentLabels[grp.groupIdx];
      if (!lbl) return;
      const card = document.createElement('div');
      card.className = 'im-card';
      const ru = window._renderMap && window._renderMap[lbl.cluster_id];
      card.innerHTML = `
        <div class="im-thumb" style="border:2px solid ${lbl.color}44">
          ${ru ? `<img src="${ru}" style="width:100%;height:100%;object-fit:cover;border-radius:4px"/>`
               : `<div style="width:100%;height:100%;background:${lbl.color};border-radius:4px;
                  display:flex;align-items:center;justify-content:center;font-size:20px;">◈</div>`}
        </div>
        <div class="im-label" style="color:${lbl.color}">${lbl.label.replace(/_/g,' ')}</div>`;
      card.onclick = () => { _isolateGroup(grp.groupIdx); closeIsolate(); };
      card.addEventListener('mouseenter', () => { highlightCluster(lbl.cluster_id); });
      card.addEventListener('mouseleave', () => { highlightCluster(null); });
      grid.appendChild(card);
    });
  }

  window.isolateCluster = function(cid) {
    const g = meshGroups.find(x => x.clusterId === cid);
    if (g) _isolateGroup(g.groupIdx);
    closeIsolate();
  };

  function _isolateGroup(ti) {
    viewMode = 'isolate';
    document.querySelectorAll('.tb-btn').forEach(b => b.classList.remove('active'));
    const btn = document.getElementById('btn-isolate');
    if (btn) btn.classList.add('active');
    const ec = document.getElementById('explodeCtrl');
    if (ec) ec.style.display = 'none';
    _resetExplode();

    meshGroups.forEach(grp => {
      const isTarget = grp.groupIdx === ti;
      grp.meshes.forEach(mesh => {
        mesh.visible = true;
        if (mesh.material) {
          mesh.material.transparent     = !isTarget;
          mesh.material.opacity         = isTarget ? 1.0 : 0.04;
          mesh.material.depthWrite      = isTarget;
          if (!mesh.geometry.attributes.color)
            mesh.material.emissiveIntensity = isTarget ? 0.30 : 0.0;
        }
      });
    });

    // Smooth camera fly-to for the isolated part
    const tg = meshGroups.find(g => g.groupIdx === ti);
    if (tg && tg.meshes.length) {
      tg.meshes[0].updateMatrixWorld(true);
      const box = new THREE.Box3();
      tg.meshes.forEach(m => box.expandByObject(m));
      if (!box.isEmpty()) {
        const cen = new THREE.Vector3(); box.getCenter(cen);
        const siz = new THREE.Vector3(); box.getSize(siz);
        const d   = Math.max(siz.x, siz.y, siz.z, 0.3) * 3.5;
        // Animate camera
        const startPos  = camera.position.clone();
        const targetPos = new THREE.Vector3(cen.x + d*0.15, cen.y + d*0.25, cen.z + d);
        const startTgt  = controls.target.clone();
        let t = 0;
        const fly = () => {
          t = Math.min(1, t + 0.05);
          const ease = 1 - Math.pow(1 - t, 3);
          camera.position.lerpVectors(startPos, targetPos, ease);
          controls.target.lerpVectors(startTgt, cen, ease);
          controls.update();
          if (t < 1) requestAnimationFrame(fly);
        };
        fly();
      }
    }
  }

  window.highlightCluster = function(cid) {
    meshGroups.forEach(g=>g.meshes.forEach(m=>{
      if (m.material&&!m.geometry.attributes.color)
        m.material.emissiveIntensity = cid===null?0.06:(g.clusterId===cid?0.5:0.01);
    }));
  };

  window.toggleWireframe = function() {
    wireframeOn=!wireframeOn;
    meshGroups.forEach(g=>g.meshes.forEach(m=>{if(m.material)m.material.wireframe=wireframeOn;}));
    const btn=document.getElementById('btn-wire'); if(btn) btn.classList.toggle('active',wireframeOn);
  };

  window.toggleLabels = function() {
    labelsVisible=!labelsVisible;
    const l=document.getElementById('labelLayer'); if(l) l.style.display=labelsVisible?'':'none';
    const btn=document.getElementById('btn-lbl'); if(btn) btn.classList.toggle('active',labelsVisible);
  };

  window.resetCamera = function() { if(rootGroup) _fitCamera(rootGroup); };

  function _fitCamera(obj) {
    // Force world matrix update so Box3 is accurate
    obj.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(obj);
    if (box.isEmpty()) {
      // Fallback camera position if object is empty/degenerate
      camera.position.set(0, 2, 6); controls.target.set(0,0,0); controls.update(); return;
    }
    const cen = new THREE.Vector3(); box.getCenter(cen);
    const siz = new THREE.Vector3(); box.getSize(siz);
    const d   = Math.max(siz.x, siz.y, siz.z, 1) * 2.2;
    camera.position.set(cen.x + d*0.2, cen.y + d*0.4, cen.z + d);
    camera.lookAt(cen); controls.target.copy(cen); controls.update();
  }

  function _hexToKey255(hex) {
    const h=hex.replace('#','');
    return parseInt(h.slice(0,2),16)+','+parseInt(h.slice(2,4),16)+','+parseInt(h.slice(4,6),16);
  }

  function _clearScene() {
    if (rootGroup) { scene.remove(rootGroup); rootGroup=null; }
    meshGroups=[]; labelSprites=[]; currentLabels=[];
    const l=document.getElementById('labelLayer'); if(l) l.innerHTML='';
  }

  window.clearScene = _clearScene;
  init();
})();
