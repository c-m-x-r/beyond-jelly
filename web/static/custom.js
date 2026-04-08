// Gene definitions
const GENES = [
    { name: 'CP1 X',          idx: 0 },
    { name: 'CP1 Y',          idx: 1 },
    { name: 'CP2 X',          idx: 2 },
    { name: 'CP2 Y',          idx: 3 },
    { name: 'Bell Tip X',     idx: 4 },
    { name: 'Bell Tip Y',     idx: 5 },
    { name: 'Thickness Base', idx: 6 },
    { name: 'Thickness Mid',  idx: 7 },
    { name: 'Thickness Tip',  idx: 8 },
];

// State
let customGenome = [];
let bounds = { lower: [], upper: [], default: [] };
let renderTimeout = null;
let jellyColor = '#4ECDC4';
let muscleColor = '#FF6B6B';

// === Init ===

async function init() {
    const resp = await fetch(API_BASE + '/api/bounds');
    bounds = await resp.json();
    customGenome = [...bounds.default];

    buildGeneControls();
    initBezierOverlay();
    updateBezierOverlay();
    renderMorphology();
    loadAquarium();

    document.getElementById('btn-randomize').addEventListener('click', randomize);

    document.getElementById('btn-step1-next').addEventListener('click', step1Next);
    document.getElementById('btn-step2-back').addEventListener('click', () => goToStep(1));
    document.getElementById('btn-submit').addEventListener('click', submitJellyfish);
    document.getElementById('btn-design-another').addEventListener('click', designAnother);

    document.getElementById('jelly-color').addEventListener('input', onJellyColorChange);
    document.getElementById('muscle-color').addEventListener('input', onMuscleColorChange);
}

// === Wizard ===

function goToStep(n) {
    document.querySelectorAll('.wizard-step').forEach(el => el.classList.remove('active'));
    document.getElementById(`step-${n}`).classList.add('active');
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function step1Next() {
    const name = document.getElementById('jelly-name').value.trim();
    if (!name) {
        const hint = document.getElementById('name-hint');
        hint.textContent = 'Give your jellyfish a name first.';
        document.getElementById('jelly-name').focus();
        return;
    }
    document.getElementById('name-hint').textContent = '';
    goToStep(2);
}

function designAnother() {
    customGenome = [...bounds.default];
    updateSliders();
    updateBezierOverlay();
    document.getElementById('jelly-name').value = '';
    document.getElementById('contact-email').value = '';
    applyColors('#4ECDC4', '#FF6B6B');
    goToStep(1);
    renderMorphology();
    loadAquarium();
}

// === Gene controls ===

function buildGeneControls() {
    const container = document.getElementById('custom-gene-controls');
    GENES.forEach(gene => {
        const control = document.createElement('div');
        control.className = 'gene-control';

        const label = document.createElement('label');
        label.innerHTML = `
            <span>${gene.name}</span>
            <span class="value" id="cval-${gene.idx}">${customGenome[gene.idx].toFixed(3)}</span>
        `;

        const slider = document.createElement('input');
        slider.type = 'range';
        slider.id = `cgene-${gene.idx}`;
        slider.min = bounds.lower[gene.idx];
        slider.max = bounds.upper[gene.idx];
        slider.step = 0.001;
        slider.value = customGenome[gene.idx];

        slider.addEventListener('input', (e) => {
            customGenome[gene.idx] = parseFloat(e.target.value);
            document.getElementById(`cval-${gene.idx}`).textContent = parseFloat(e.target.value).toFixed(3);
            updateBezierOverlay();
            debouncedRender();
        });

        control.appendChild(label);
        control.appendChild(slider);
        container.appendChild(control);
    });
}

function updateSliders() {
    GENES.forEach(gene => {
        const slider = document.getElementById(`cgene-${gene.idx}`);
        const val = document.getElementById(`cval-${gene.idx}`);
        if (slider) slider.value = customGenome[gene.idx];
        if (val) val.textContent = customGenome[gene.idx].toFixed(3);
    });
}

// === Color helpers ===

function hslToHex(h, s, l) {
    s /= 100; l /= 100;
    const a = s * Math.min(l, 1 - l);
    const f = n => {
        const k = (n + h / 30) % 12;
        const c = l - a * Math.max(Math.min(k - 3, 9 - k, 1), -1);
        return Math.round(255 * c).toString(16).padStart(2, '0');
    };
    return `#${f(0)}${f(8)}${f(4)}`;
}

function randomVibrantHex(hueShift = 0) {
    const h = (Math.floor(Math.random() * 360) + hueShift) % 360;
    const s = 60 + Math.floor(Math.random() * 30);   // 60–90%
    const l = 42 + Math.floor(Math.random() * 22);   // 42–64%
    return hslToHex(h, s, l);
}

function applyColors(jelly, muscle) {
    jellyColor = jelly;
    muscleColor = muscle;
    document.getElementById('jelly-color').value = jelly;
    document.getElementById('muscle-color').value = muscle;
    document.getElementById('jelly-color-label').textContent = jelly;
    document.getElementById('muscle-color-label').textContent = muscle;
}

// === Randomize: genome + colors ===

async function randomize() {
    const data = await (await fetch(API_BASE + '/api/random')).json();
    customGenome = data.genome;
    // Pick two hues separated by 90–180° for contrast
    const jellyHex = randomVibrantHex();
    const muscleHex = randomVibrantHex(90 + Math.floor(Math.random() * 90));
    applyColors(jellyHex, muscleHex);
    updateSliders(); updateBezierOverlay(); renderMorphology();
}

// === Rendering ===

function debouncedRender() {
    clearTimeout(renderTimeout);
    renderTimeout = setTimeout(renderMorphology, 300);
}

async function renderMorphology() {
    const img = document.getElementById('custom-morphology-img');
    const loading = document.getElementById('custom-loading');
    loading.classList.remove('hidden');
    img.classList.remove('loaded');

    try {
        const resp = await fetch(API_BASE + '/api/custom/render', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ genome: customGenome, jelly_color: jellyColor, muscle_color: muscleColor }),
        });
        const data = await resp.json();
        if (data.error) { loading.textContent = 'Error'; return; }
        img.src = data.image;
        img.onload = () => { img.classList.add('loaded'); loading.classList.add('hidden'); updateBezierOverlay(); };
    } catch {
        loading.textContent = 'Render failed';
    }
}

// === Color pickers — update main preview ===

function onJellyColorChange(e) {
    jellyColor = e.target.value;
    document.getElementById('jelly-color-label').textContent = jellyColor;
    debouncedRender();
}

function onMuscleColorChange(e) {
    muscleColor = e.target.value;
    document.getElementById('muscle-color-label').textContent = muscleColor;
    debouncedRender();
}

// === Aquarium ===

async function loadAquarium() {
    const grid = document.getElementById('aquarium-grid');
    const countEl = document.getElementById('aquarium-count');
    try {
        const data = await (await fetch(API_BASE + '/api/custom/aquarium')).json();
        const subs = data.submissions;

        grid.innerHTML = '';
        if (subs.length === 0) {
            const empty = document.createElement('div');
            empty.className = 'aquarium-empty';
            empty.textContent = 'Be the first to design!';
            grid.appendChild(empty);
            countEl.textContent = '';
            return;
        }

        countEl.textContent = `(${subs.length})`;
        subs.forEach(sub => {
            const fig = document.createElement('figure');
            fig.className = 'aquarium-item';

            const img = document.createElement('img');
            img.src = `${API_BASE}/api/custom/thumbnail/${sub.id}`;
            img.alt = sub.name;
            img.loading = 'lazy';

            const cap = document.createElement('figcaption');
            cap.textContent = sub.name;

            fig.appendChild(img);
            fig.appendChild(cap);
            grid.appendChild(fig);
        });
    } catch {
        grid.innerHTML = '<div class="aquarium-empty">Could not load aquarium.</div>';
    }
}

// === Submission ===

async function submitJellyfish() {
    const btn = document.getElementById('btn-submit');
    const name = document.getElementById('jelly-name').value.trim();
    const email = document.getElementById('contact-email').value.trim();

    btn.disabled = true;
    btn.textContent = 'Submitting...';

    try {
        const resp = await fetch(API_BASE + '/api/custom/submit', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                genome: customGenome,
                name,
                jelly_color: jellyColor,
                muscle_color: muscleColor,
                email: email || null,
            }),
        });
        const data = await resp.json();

        if (!resp.ok || data.error) {
            alert(data.error || 'Submission failed, please try again.');
            btn.disabled = false;
            btn.textContent = 'Submit My Jellyfish';
            return;
        }

        document.getElementById('result-name').textContent = name;

        const resultImg = document.getElementById('result-preview-img');
        resultImg.src = `${API_BASE}/api/custom/thumbnail/${data.id}`;
        resultImg.onerror = () => {
            fetch(API_BASE + '/api/custom/render', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ genome: customGenome, jelly_color: jellyColor, muscle_color: muscleColor }),
            }).then(r => r.json()).then(d => { if (d.image) resultImg.src = d.image; });
        };

        showPrediction(data.prediction);
        goToStep(3);
        loadAquarium();

    } catch {
        alert('Submission failed — check your connection and try again.');
        btn.disabled = false;
        btn.textContent = 'Submit My Jellyfish';
    }
}

function showPrediction(pred) {
    const content = document.getElementById('prediction-content');
    const unavailable = document.getElementById('prediction-unavailable');

    if (!pred) {
        content.classList.add('hidden');
        unavailable.classList.remove('hidden');
        return;
    }

    content.classList.remove('hidden');
    unavailable.classList.add('hidden');

    document.getElementById('pred-displacement').textContent = `~${pred.displacement.toFixed(4)} units`;
    document.getElementById('pred-percentile').textContent = `Top ${(100 - pred.percentile).toFixed(1)}%`;
    document.getElementById('pred-generation').textContent = `Generation ${pred.generation}`;
    document.getElementById('pred-total').textContent = `${pred.total_individuals.toLocaleString()} genomes`;
    document.getElementById('percentile-fill').style.width = `${Math.max(2, pred.percentile)}%`;
}

// === Bezier Overlay ===

const SPAWN_X = 0.5;
const SPAWN_Y = 0.4;
const PAYLOAD_HALF_W = 0.04;
const BZ_START_X = SPAWN_X + PAYLOAD_HALF_W;
const BZ_START_Y = SPAWN_Y;
const SVG_NS = 'http://www.w3.org/2000/svg';
let bzDrag = null;

function genomeToBezierPoints(g) {
    return {
        p0: { x: BZ_START_X,                       y: BZ_START_Y },
        p1: { x: BZ_START_X + Math.abs(g[0]),      y: BZ_START_Y + g[1] },
        p2: { x: BZ_START_X + Math.abs(g[2]),      y: BZ_START_Y + g[3] },
        p3: { x: BZ_START_X + Math.abs(g[4]),      y: BZ_START_Y + g[5] },
    };
}

function mirrorBzPoint(p) { return { x: 1 - p.x, y: p.y }; }

function bzPathD(p0, p1, p2, p3) {
    return `M ${p0.x} ${p0.y} C ${p1.x} ${p1.y} ${p2.x} ${p2.y} ${p3.x} ${p3.y}`;
}

function initBezierOverlay() {
    const svg = document.getElementById('custom-bezier-overlay');
    if (!svg) return;

    const g = document.createElementNS(SVG_NS, 'g');
    g.id = 'cbz-group';
    g.setAttribute('transform', 'scale(1,-1) translate(0,-1)');
    svg.appendChild(g);

    const makePath = (id, stroke, sw, dash, op) => {
        const el = document.createElementNS(SVG_NS, 'path');
        el.id = id; el.setAttribute('fill', 'none'); el.setAttribute('stroke', stroke);
        el.setAttribute('stroke-width', sw);
        if (dash) el.setAttribute('stroke-dasharray', dash);
        el.setAttribute('opacity', op); el.setAttribute('pointer-events', 'none');
        return el;
    };

    const makeLine = (id, stroke, sw, dash) => {
        const el = document.createElementNS(SVG_NS, 'line');
        el.id = id; el.setAttribute('stroke', stroke); el.setAttribute('stroke-width', sw);
        if (dash) el.setAttribute('stroke-dasharray', dash);
        el.setAttribute('pointer-events', 'none');
        return el;
    };

    g.appendChild(makePath('cbz-spine-l', '#1A1A1A', '0.004', '0.015 0.008', '0.3'));
    g.appendChild(makePath('cbz-spine-r', '#1A1A1A', '0.005', '0.015 0.008', '0.65'));
    g.appendChild(makeLine('cbz-ctrl-01', '#666', '0.003', '0.008 0.005'));
    g.appendChild(makeLine('cbz-ctrl-32', '#666', '0.003', '0.008 0.005'));

    const p0el = document.createElementNS(SVG_NS, 'circle');
    p0el.id = 'cbz-p0'; p0el.setAttribute('r', '0.012');
    p0el.setAttribute('fill', 'rgba(26,26,26,0.7)'); p0el.setAttribute('stroke', 'white');
    p0el.setAttribute('stroke-width', '0.004'); p0el.setAttribute('pointer-events', 'none');
    g.appendChild(p0el);

    [
        { id: 'cbz-p1', geneX: 0, geneY: 1, color: 'rgba(90,180,175,0.7)' },
        { id: 'cbz-p2', geneX: 2, geneY: 3, color: 'rgba(200,110,110,0.7)' },
        { id: 'cbz-p3', geneX: 4, geneY: 5, color: 'rgba(195,155,80,0.7)' },
    ].forEach(({ id, geneX, geneY, color }) => {
        const h = document.createElementNS(SVG_NS, 'circle');
        h.id = id; h.setAttribute('r', '0.013');
        h.setAttribute('fill', color); h.setAttribute('stroke', 'rgba(255,255,255,0.8)');
        h.setAttribute('stroke-width', '0.004');
        h.classList.add('bz-handle');
        h.dataset.geneX = geneX; h.dataset.geneY = geneY;
        h.addEventListener('mousedown', onBzHandleMouseDown);
        g.appendChild(h);
    });
}

function updateBezierOverlay() {
    const svg = document.getElementById('custom-bezier-overlay');
    if (!svg || !svg.querySelector('#cbz-group')) return;

    const pts = genomeToBezierPoints(customGenome);
    const lpts = { p0: mirrorBzPoint(pts.p0), p1: mirrorBzPoint(pts.p1), p2: mirrorBzPoint(pts.p2), p3: mirrorBzPoint(pts.p3) };

    document.getElementById('cbz-spine-r').setAttribute('d', bzPathD(pts.p0, pts.p1, pts.p2, pts.p3));
    document.getElementById('cbz-spine-l').setAttribute('d', bzPathD(lpts.p0, lpts.p1, lpts.p2, lpts.p3));

    const c01 = document.getElementById('cbz-ctrl-01');
    c01.setAttribute('x1', pts.p0.x); c01.setAttribute('y1', pts.p0.y);
    c01.setAttribute('x2', pts.p1.x); c01.setAttribute('y2', pts.p1.y);

    const c32 = document.getElementById('cbz-ctrl-32');
    c32.setAttribute('x1', pts.p3.x); c32.setAttribute('y1', pts.p3.y);
    c32.setAttribute('x2', pts.p2.x); c32.setAttribute('y2', pts.p2.y);

    [['cbz-p0', pts.p0], ['cbz-p1', pts.p1], ['cbz-p2', pts.p2], ['cbz-p3', pts.p3]].forEach(([id, p]) => {
        document.getElementById(id).setAttribute('cx', p.x);
        document.getElementById(id).setAttribute('cy', p.y);
    });
}

function onBzHandleMouseDown(e) {
    e.preventDefault();
    bzDrag = { geneX: parseInt(e.target.dataset.geneX), geneY: parseInt(e.target.dataset.geneY), handleEl: e.target };
    e.target.classList.add('dragging');
    document.addEventListener('mousemove', onBzDragMove);
    document.addEventListener('mouseup', onBzDragEnd);
}

function onBzDragMove(e) {
    if (!bzDrag) return;
    const svg = document.getElementById('custom-bezier-overlay');
    const pt = svg.createSVGPoint();
    pt.x = e.clientX; pt.y = e.clientY;
    const sp = pt.matrixTransform(svg.getScreenCTM().inverse());
    const { geneX, geneY } = bzDrag;
    customGenome[geneX] = Math.max(bounds.lower[geneX], Math.min(bounds.upper[geneX], sp.x - BZ_START_X));
    customGenome[geneY] = Math.max(bounds.lower[geneY], Math.min(bounds.upper[geneY], (1 - sp.y) - BZ_START_Y));
    updateBezierOverlay();
    updateSliders();
}

function onBzDragEnd() {
    if (!bzDrag) return;
    bzDrag.handleEl.classList.remove('dragging');
    bzDrag = null;
    document.removeEventListener('mousemove', onBzDragMove);
    document.removeEventListener('mouseup', onBzDragEnd);
    renderMorphology();
}

init();
