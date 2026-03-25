// Gene definitions (11D genome: 9 morphology + 2 actuation timing)
const GENES = [
    { name: 'CP1 X', key: 'cp1_x', idx: 0 },
    { name: 'CP1 Y', key: 'cp1_y', idx: 1 },
    { name: 'CP2 X', key: 'cp2_x', idx: 2 },
    { name: 'CP2 Y', key: 'cp2_y', idx: 3 },
    { name: 'Bell Tip X', key: 'tip_x', idx: 4 },
    { name: 'Bell Tip Y', key: 'tip_y', idx: 5 },
    { name: 'Thickness Base', key: 't_base', idx: 6 },
    { name: 'Thickness Mid', key: 't_mid', idx: 7 },
    { name: 'Thickness Tip', key: 't_tip', idx: 8 },
    { name: 'Contraction Frac', key: 'act_contraction', idx: 9 },
    { name: 'Refractory Frac', key: 'act_refractory', idx: 10 },
];

// State
let currentGenome = [];
let bounds = { lower: [], upper: [], default: [] };
let renderTimeout = null;
let evoData = { generations: [], currentGen: null, individuals: [], currentLog: 'evolution_log.csv' };

// Initialize
async function init() {
    // Fetch bounds
    const response = await fetch('/api/bounds');
    bounds = await response.json();

    // Set initial genome
    currentGenome = [...bounds.default];

    // Build UI
    buildGeneControls();
    updateGenomeDisplay();

    // Init Bezier overlay
    initBezierOverlay();
    updateBezierOverlay();

    // Initial render
    renderMorphology();

    // Event listeners
    document.getElementById('btn-default').addEventListener('click', loadDefault);
    document.getElementById('btn-aurelia').addEventListener('click', loadAurelia);
    document.getElementById('btn-random').addEventListener('click', loadRandom);
    document.getElementById('btn-load').addEventListener('click', loadFromJson);
    document.getElementById('btn-copy').addEventListener('click', copyToClipboard);

    // Evolution browser
    document.getElementById('log-select').addEventListener('change', onLogSelect);
    document.getElementById('gen-select').addEventListener('change', onGenSelect);
    document.getElementById('btn-render-gen').addEventListener('click', renderGeneration);
    document.getElementById('btn-sim-gen').addEventListener('click', simulateGeneration);
    loadEvolutionLogs();
}

// Build gene control sliders
function buildGeneControls() {
    const container = document.getElementById('gene-controls');

    GENES.forEach(gene => {
        const control = document.createElement('div');
        control.className = 'gene-control';

        const label = document.createElement('label');
        label.innerHTML = `
            <span>${gene.name}</span>
            <span class="value" id="value-${gene.idx}">${currentGenome[gene.idx].toFixed(3)}</span>
        `;

        const slider = document.createElement('input');
        slider.type = 'range';
        slider.id = `gene-${gene.idx}`;
        slider.min = bounds.lower[gene.idx];
        slider.max = bounds.upper[gene.idx];
        slider.step = 0.001;
        slider.value = currentGenome[gene.idx];

        slider.addEventListener('input', (e) => {
            currentGenome[gene.idx] = parseFloat(e.target.value);
            document.getElementById(`value-${gene.idx}`).textContent = e.target.value;
            updateGenomeDisplay();
            updateBezierOverlay();
            debouncedRender();
        });

        control.appendChild(label);
        control.appendChild(slider);
        container.appendChild(control);
    });
}

// Update all slider values
function updateSliders() {
    GENES.forEach(gene => {
        const slider = document.getElementById(`gene-${gene.idx}`);
        const valueDisplay = document.getElementById(`value-${gene.idx}`);
        slider.value = currentGenome[gene.idx];
        valueDisplay.textContent = currentGenome[gene.idx].toFixed(3);
    });
}

// Update genome JSON display
function updateGenomeDisplay() {
    const textarea = document.getElementById('genome-json');
    textarea.value = JSON.stringify(currentGenome.map(v => parseFloat(v.toFixed(3))));
}

// Debounced render (300ms delay)
function debouncedRender() {
    clearTimeout(renderTimeout);
    renderTimeout = setTimeout(renderMorphology, 300);
}

// Render morphology via API
async function renderMorphology() {
    const img = document.getElementById('morphology-img');
    const loading = document.getElementById('loading');

    // Show loading
    loading.classList.remove('hidden');
    img.classList.remove('loaded');

    try {
        const response = await fetch('/api/render', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ genome: currentGenome })
        });

        const data = await response.json();

        if (data.error) {
            console.error('Render error:', data.error);
            return;
        }

        // Update image
        img.src = data.image;
        img.onload = () => {
            img.classList.add('loaded');
            loading.classList.add('hidden');
            updateBezierOverlay();
        };

        // Update stats
        updateStats(data.stats);

    } catch (error) {
        console.error('API error:', error);
        loading.textContent = 'Render failed';
    }
}

// Update statistics panel
function updateStats(stats) {
    document.getElementById('stat-total').textContent = stats.n_total.toLocaleString();
    document.getElementById('stat-robot').textContent = `${stats.n_robot.toLocaleString()} (${(stats.n_robot / stats.n_total * 100).toFixed(1)}%)`;
    document.getElementById('stat-jelly').textContent = stats.n_jelly.toLocaleString();
    document.getElementById('stat-muscle').textContent = stats.muscle_count.toLocaleString();
    document.getElementById('stat-water').textContent = stats.n_water.toLocaleString();

    const geomEl = document.getElementById('stat-geometry');
    if (stats.self_intersecting) {
        geomEl.textContent = 'INVALID';
        geomEl.style.color = 'var(--highlight)';
    } else {
        geomEl.textContent = 'Valid';
        geomEl.style.color = 'var(--accent)';
    }
}

// Load default genome
async function loadDefault() {
    const response = await fetch('/api/default');
    const data = await response.json();
    currentGenome = data.genome;
    updateSliders();
    updateGenomeDisplay();
    renderMorphology();
}

// Load Aurelia genome
async function loadAurelia() {
    const response = await fetch('/api/aurelia');
    const data = await response.json();
    currentGenome = data.genome;
    updateSliders();
    updateGenomeDisplay();
    renderMorphology();
}

// Load random genome
async function loadRandom() {
    const response = await fetch('/api/random');
    const data = await response.json();
    currentGenome = data.genome;
    updateSliders();
    updateGenomeDisplay();
    renderMorphology();
}

// Load genome from JSON textarea
function loadFromJson() {
    try {
        const textarea = document.getElementById('genome-json');
        const genome = JSON.parse(textarea.value);

        if (!Array.isArray(genome) || (genome.length !== 9 && genome.length !== 11)) {
            alert('Invalid genome: must be array of 9 or 11 numbers');
            return;
        }

        // Validate bounds for however many genes are present
        for (let i = 0; i < genome.length; i++) {
            if (genome[i] < bounds.lower[i] || genome[i] > bounds.upper[i]) {
                alert(`Gene ${i} out of bounds: ${genome[i]} not in [${bounds.lower[i]}, ${bounds.upper[i]}]`);
                return;
            }
        }

        currentGenome = genome;
        updateSliders();
        updateGenomeDisplay();
        renderMorphology();

    } catch (error) {
        alert('Invalid JSON: ' + error.message);
    }
}

// Copy genome to clipboard
async function copyToClipboard() {
    const textarea = document.getElementById('genome-json');
    try {
        await navigator.clipboard.writeText(textarea.value);
        const btn = document.getElementById('btn-copy');
        const originalText = btn.textContent;
        btn.textContent = 'Copied!';
        setTimeout(() => {
            btn.textContent = originalText;
        }, 1500);
    } catch (error) {
        // Fallback for older browsers
        textarea.select();
        document.execCommand('copy');
    }
}

// === Evolution Browser ===

async function loadEvolutionLogs() {
    try {
        const response = await fetch('/api/evolution/logs');
        const data = await response.json();
        const select = document.getElementById('log-select');

        if (data.logs.length === 0) {
            select.innerHTML = '<option value="">No logs found</option>';
            document.getElementById('evo-status').textContent = 'No evolution logs found';
            return;
        }

        select.disabled = false;
        select.innerHTML = '';
        data.logs.forEach(name => {
            const opt = document.createElement('option');
            opt.value = name;
            opt.textContent = name;
            if (name === 'evolution_log.csv') opt.selected = true;
            select.appendChild(opt);
        });

        evoData.currentLog = select.value;
        loadEvolutionSummary();
    } catch (error) {
        document.getElementById('evo-status').textContent = 'Failed to load log list';
    }
}

function onLogSelect() {
    const logName = document.getElementById('log-select').value;
    if (!logName) return;
    evoData.currentLog = logName;

    // Reset generation state
    evoData.currentGen = null;
    evoData.individuals = [];
    document.getElementById('evo-tbody').innerHTML =
        '<tr><td colspan="7" class="evo-empty">Select a generation above</td></tr>';
    document.getElementById('evo-grid-container').classList.add('hidden');

    loadEvolutionSummary();
}

async function loadEvolutionSummary() {
    try {
        const response = await fetch(`/api/evolution/summary?log=${encodeURIComponent(evoData.currentLog)}`);
        const data = await response.json();
        evoData.generations = data.generations;

        const select = document.getElementById('gen-select');
        const btnRender = document.getElementById('btn-render-gen');
        const btnSim = document.getElementById('btn-sim-gen');
        const status = document.getElementById('evo-status');

        if (data.generations.length === 0) {
            select.disabled = true;
            btnRender.disabled = true;
            btnSim.disabled = true;
            select.innerHTML = '<option value="">No data</option>';
            status.textContent = `${evoData.currentLog}: no data`;
            return;
        }

        select.disabled = false;
        btnRender.disabled = false;
        btnSim.disabled = false;
        select.innerHTML = '<option value="">Select generation...</option>';

        data.generations.forEach(g => {
            const opt = document.createElement('option');
            opt.value = g.generation;
            opt.textContent = `Gen ${g.generation} — best: ${g.best_fitness.toFixed(3)}, avg: ${g.avg_fitness.toFixed(3)} (${g.count} ind.)`;
            select.appendChild(opt);
        });

        status.textContent = `${evoData.currentLog}: ${data.generations.length} generations, ${data.total_individuals} individuals`;
        drawConvergenceChart();
    } catch (error) {
        document.getElementById('evo-status').textContent = 'Failed to load evolution data';
    }
}

async function onGenSelect() {
    const gen = parseInt(document.getElementById('gen-select').value);
    if (isNaN(gen)) {
        document.getElementById('evo-tbody').innerHTML =
            '<tr><td colspan="7" class="evo-empty">Select a generation above</td></tr>';
        evoData.currentGen = null;
        evoData.individuals = [];
        return;
    }

    try {
        const response = await fetch(`/api/evolution/generation/${gen}?log=${encodeURIComponent(evoData.currentLog)}`);
        const data = await response.json();
        evoData.currentGen = gen;
        evoData.individuals = data.individuals;
        populateIndividualTable(data.individuals);
        drawConvergenceChart();
    } catch (error) {
        document.getElementById('evo-tbody').innerHTML =
            '<tr><td colspan="7" class="evo-empty">Failed to load generation</td></tr>';
    }
}

function populateIndividualTable(individuals) {
    const tbody = document.getElementById('evo-tbody');
    tbody.innerHTML = '';

    individuals.forEach((ind, idx) => {
        const tr = document.createElement('tr');
        if (idx === 0) tr.classList.add('best');

        tr.innerHTML = `
            <td class="mono">${ind.individual}</td>
            <td class="mono">${ind.fitness.toFixed(4)}</td>
            <td class="mono">${ind.final_y.toFixed(4)}</td>
            <td class="mono">${ind.drift.toFixed(4)}</td>
            <td class="mono">${ind.muscle_count}</td>
            <td>${ind.valid ? 'Y' : 'N'}</td>
            <td><button class="btn-load-ind" data-idx="${idx}">Load</button></td>
        `;
        tbody.appendChild(tr);
    });

    // Attach click handlers
    tbody.querySelectorAll('.btn-load-ind').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const idx = parseInt(e.target.dataset.idx);
            loadIndividual(individuals[idx]);
        });
    });
}

function loadIndividual(ind) {
    currentGenome = [...ind.genome];
    // Pad 9-gene (legacy) genomes with defaults for timing genes 9 & 10
    while (currentGenome.length < GENES.length) {
        currentGenome.push(bounds.default[currentGenome.length]);
    }
    updateSliders();
    updateGenomeDisplay();
    renderMorphology();

    // Scroll to viewer
    document.querySelector('.viewer').scrollIntoView({ behavior: 'smooth' });
}

async function renderGeneration() {
    if (!evoData.individuals.length) return;

    const container = document.getElementById('evo-grid-container');
    const loading = document.getElementById('evo-grid-loading');
    const img = document.getElementById('evo-grid-img');
    const btn = document.getElementById('btn-render-gen');

    container.classList.remove('hidden');
    loading.classList.remove('hidden');
    img.style.opacity = '0';
    btn.disabled = true;
    btn.textContent = 'Rendering...';

    try {
        const response = await fetch('/api/render/grid', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ individuals: evoData.individuals })
        });

        const data = await response.json();

        if (data.error) {
            loading.textContent = 'Render failed: ' + data.error;
            return;
        }

        img.src = data.image;
        img.onload = () => {
            img.style.opacity = '1';
            loading.classList.add('hidden');
        };
    } catch (error) {
        loading.textContent = 'Render failed';
    } finally {
        btn.disabled = false;
        btn.textContent = 'Render Generation';
    }
}

// === Simulate Generation ===

let simPollInterval = null;

async function simulateGeneration() {
    if (evoData.currentGen === null) return;

    const btn = document.getElementById('btn-sim-gen');
    const progress = document.getElementById('sim-progress');
    const progressText = document.getElementById('sim-progress-text');
    const progressFill = document.getElementById('sim-progress-fill');

    btn.disabled = true;
    btn.textContent = 'Starting...';
    progress.classList.remove('hidden');
    progressText.textContent = 'Starting simulation (GPU init may take a moment)...';
    progressFill.style.width = '0%';

    // Hide previous video
    document.getElementById('evo-video-container').classList.add('hidden');

    try {
        const response = await fetch('/api/simulate/generation', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                generation: evoData.currentGen,
                log: evoData.currentLog,
                frames: parseInt(document.getElementById('sim-frames').value),
                web_palette: true,
            })
        });

        const data = await response.json();

        if (data.error) {
            progressText.textContent = 'Error: ' + data.error;
            btn.disabled = false;
            btn.textContent = 'Simulate Generation';
            return;
        }

        // Start polling for progress
        pollSimStatus(data.video, data.total_frames);

    } catch (error) {
        progressText.textContent = 'Failed to start simulation';
        btn.disabled = false;
        btn.textContent = 'Simulate Generation';
    }
}

function pollSimStatus(expectedVideo, totalFrames) {
    if (simPollInterval) clearInterval(simPollInterval);

    const progressText = document.getElementById('sim-progress-text');
    const progressFill = document.getElementById('sim-progress-fill');
    const btn = document.getElementById('btn-sim-gen');

    simPollInterval = setInterval(async () => {
        try {
            const response = await fetch('/api/simulate/status');
            const status = await response.json();

            if (status.state === 'running' || status.state === 'starting') {
                const frame = status.frame || 0;
                const total = status.total_frames || totalFrames;
                const pct = total > 0 ? (frame / total * 100) : 0;
                progressText.textContent = `Rendering: ${frame}/${total} frames`;
                progressFill.style.width = pct + '%';

            } else if (status.state === 'done') {
                clearInterval(simPollInterval);
                simPollInterval = null;

                progressText.textContent = 'Simulation complete';
                progressFill.style.width = '100%';
                btn.disabled = false;
                btn.textContent = 'Simulate Generation';

                // Show video
                const videoUrl = `/api/simulate/video/${encodeURIComponent(expectedVideo)}`;
                showSimVideo(videoUrl, expectedVideo);

                // Hide progress after a moment
                setTimeout(() => {
                    document.getElementById('sim-progress').classList.add('hidden');
                }, 2000);

            } else if (status.state === 'error') {
                clearInterval(simPollInterval);
                simPollInterval = null;
                progressText.textContent = 'Simulation failed: ' + (status.error || 'unknown error');
                btn.disabled = false;
                btn.textContent = 'Simulate Generation';
            }
        } catch (error) {
            // Network error, keep polling
        }
    }, 1500);
}

function showSimVideo(videoUrl, filename) {
    const container = document.getElementById('evo-video-container');
    const video = document.getElementById('evo-video');
    const source = document.getElementById('evo-video-src');
    const label = document.getElementById('evo-video-label');
    const link = document.getElementById('evo-video-link');

    // Force reload by updating src with cache-bust
    const bustUrl = videoUrl + '?t=' + Date.now();
    source.src = bustUrl;
    video.load();

    label.textContent = filename;
    link.href = bustUrl;
    container.classList.remove('hidden');

    container.scrollIntoView({ behavior: 'smooth' });
}

// === Convergence Chart ===

function drawConvergenceChart() {
    const wrap = document.getElementById('convergence-chart-wrap');
    const canvas = document.getElementById('convergence-chart');
    const gens = evoData.generations;

    if (!gens || gens.length === 0) {
        wrap.classList.add('hidden');
        return;
    }
    wrap.classList.remove('hidden');

    // Size canvas to its CSS pixel size
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.offsetWidth || 800;
    const cssH = 200;
    canvas.width = cssW * dpr;
    canvas.height = cssH * dpr;
    canvas.style.height = cssH + 'px';

    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);

    const ml = 52, mr = 52, mt = 18, mb = 36;
    const pw = cssW - ml - mr;
    const ph = cssH - mt - mb;

    // Data
    const genNums   = gens.map(g => g.generation);
    const bestFit   = gens.map(g => g.best_fitness);
    const avgFit    = gens.map(g => g.avg_fitness);
    const sigmas    = gens.map(g => g.sigma);
    const n = gens.length;

    // Y ranges
    const fitMin = Math.min(...bestFit, ...avgFit);
    const fitMax = Math.max(...bestFit, ...avgFit);
    const fitPad = (fitMax - fitMin) * 0.1 || 0.05;
    const yFitLo = fitMin - fitPad;
    const yFitHi = fitMax + fitPad;

    const sigMin = Math.min(...sigmas);
    const sigMax = Math.max(...sigmas);
    const sigPad = (sigMax - sigMin) * 0.1 || 0.01;
    const ySigLo = sigMin - sigPad;
    const ySigHi = sigMax + sigPad;

    const xMin = genNums[0];
    const xMax = genNums[n - 1];
    const xSpan = xMax - xMin || 1;

    function toX(g)   { return ml + (g - xMin) / xSpan * pw; }
    function toYfit(v) { return mt + ph - (v - yFitLo) / (yFitHi - yFitLo) * ph; }
    function toYsig(v) { return mt + ph - (v - ySigLo) / (ySigHi - ySigLo) * ph; }

    // ── Background ──
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, cssW, cssH);

    // ── Grid lines (fitness axis) ──
    ctx.strokeStyle = '#EBEBEB';
    ctx.lineWidth = 1;
    const nGridY = 4;
    for (let i = 0; i <= nGridY; i++) {
        const v = yFitLo + (yFitHi - yFitLo) * i / nGridY;
        const y = toYfit(v);
        ctx.beginPath(); ctx.moveTo(ml, y); ctx.lineTo(ml + pw, y); ctx.stroke();
    }

    // ── Selected generation marker ──
    if (evoData.currentGen !== null) {
        const xMark = toX(evoData.currentGen);
        ctx.strokeStyle = 'rgba(26,26,26,0.15)';
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(xMark, mt); ctx.lineTo(xMark, mt + ph); ctx.stroke();
        ctx.setLineDash([]);
    }

    // ── Avg fitness (light dashed) ──
    ctx.strokeStyle = 'rgba(26,26,26,0.25)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    gens.forEach((g, i) => {
        const x = toX(g.generation), y = toYfit(g.avg_fitness);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);

    // ── Best fitness (solid dark) ──
    ctx.strokeStyle = '#1A1A1A';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    gens.forEach((g, i) => {
        const x = toX(g.generation), y = toYfit(g.best_fitness);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();

    // ── Sigma (teal, right axis) ──
    ctx.strokeStyle = '#4ECDC4';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    gens.forEach((g, i) => {
        const x = toX(g.generation), y = toYsig(g.sigma);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();

    // ── Axes ──
    ctx.strokeStyle = '#CCCCCC';
    ctx.lineWidth = 1;
    // left
    ctx.beginPath(); ctx.moveTo(ml, mt); ctx.lineTo(ml, mt + ph); ctx.stroke();
    // right
    ctx.beginPath(); ctx.moveTo(ml + pw, mt); ctx.lineTo(ml + pw, mt + ph); ctx.stroke();
    // bottom
    ctx.beginPath(); ctx.moveTo(ml, mt + ph); ctx.lineTo(ml + pw, mt + ph); ctx.stroke();

    // ── Tick labels ──
    ctx.font = `${10 * dpr / dpr}px var(--font-mono, monospace)`;
    ctx.textBaseline = 'middle';

    // Left axis ticks (fitness)
    ctx.fillStyle = '#757575';
    ctx.textAlign = 'right';
    for (let i = 0; i <= nGridY; i++) {
        const v = yFitLo + (yFitHi - yFitLo) * i / nGridY;
        const y = toYfit(v);
        ctx.fillText(v.toFixed(3), ml - 5, y);
    }

    // Right axis ticks (sigma)
    ctx.fillStyle = '#4ECDC4';
    ctx.textAlign = 'left';
    const nSigTicks = 3;
    for (let i = 0; i <= nSigTicks; i++) {
        const v = ySigLo + (ySigHi - ySigLo) * i / nSigTicks;
        const y = toYsig(v);
        ctx.fillText(v.toFixed(3), ml + pw + 5, y);
    }

    // X axis ticks (generations)
    ctx.fillStyle = '#757575';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const nXTicks = Math.min(n - 1, 8);
    const xTickStep = Math.max(1, Math.round(xSpan / nXTicks));
    for (let g = xMin; g <= xMax; g += xTickStep) {
        const x = toX(g);
        ctx.fillText(g, x, mt + ph + 5);
    }
    if (xMax % xTickStep !== 0) ctx.fillText(xMax, toX(xMax), mt + ph + 5);

    // Axis labels
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'center';
    ctx.fillStyle = '#757575';
    ctx.save();
    ctx.translate(11, mt + ph / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('fitness', 0, 0);
    ctx.restore();

    ctx.fillStyle = '#4ECDC4';
    ctx.save();
    ctx.translate(cssW - 10, mt + ph / 2);
    ctx.rotate(Math.PI / 2);
    ctx.fillText('sigma', 0, 0);
    ctx.restore();

    // Legend
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    const lx = ml + 8, ly = mt + 10;
    ctx.strokeStyle = '#1A1A1A'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx + 18, ly); ctx.stroke();
    ctx.fillStyle = '#1A1A1A'; ctx.fillText('best', lx + 22, ly);

    ctx.strokeStyle = 'rgba(26,26,26,0.25)'; ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(lx + 60, ly); ctx.lineTo(lx + 78, ly); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#757575'; ctx.fillText('avg', lx + 82, ly);

    ctx.strokeStyle = '#4ECDC4'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(lx + 118, ly); ctx.lineTo(lx + 136, ly); ctx.stroke();
    ctx.fillStyle = '#4ECDC4'; ctx.fillText('σ', lx + 140, ly);
}

// === Bezier Overlay ===
// Constants matching make_jelly.py geometry
const SPAWN_X = 0.5;
const SPAWN_Y = 0.4;
const PAYLOAD_HALF_W = 0.04;  // PAYLOAD_WIDTH / 2
const BZ_START_X = SPAWN_X + PAYLOAD_HALF_W;  // 0.54
const BZ_START_Y = SPAWN_Y;                    // 0.40
const SVG_NS = 'http://www.w3.org/2000/svg';

// Drag state
let bzDrag = null;  // { geneX, geneY, handleEl }

function genomeToBezierPoints(genome) {
    return {
        p0: { x: BZ_START_X, y: BZ_START_Y },
        p1: { x: BZ_START_X + Math.abs(genome[0]), y: BZ_START_Y + genome[1] },
        p2: { x: BZ_START_X + Math.abs(genome[2]), y: BZ_START_Y + genome[3] },
        p3: { x: BZ_START_X + Math.abs(genome[4]), y: BZ_START_Y + genome[5] },
    };
}

function mirrorBzPoint(p) {
    // Mirror around x=0.5 (spawn x): x → 1 - x
    return { x: 1 - p.x, y: p.y };
}

function bzPathD(p0, p1, p2, p3) {
    return `M ${p0.x} ${p0.y} C ${p1.x} ${p1.y} ${p2.x} ${p2.y} ${p3.x} ${p3.y}`;
}

function initBezierOverlay() {
    const svg = document.getElementById('bezier-overlay');
    if (!svg) return;

    // Group with y-flip: elements use sim coords (y=0 at bottom)
    // scale(1,-1) translate(0,-1) maps (x,y) → (x, 1-y) in SVG viewBox
    const g = document.createElementNS(SVG_NS, 'g');
    g.id = 'bz-group';
    g.setAttribute('transform', 'scale(1,-1) translate(0,-1)');
    svg.appendChild(g);

    function makePath(id, stroke, strokeWidth, dashArray, opacity) {
        const el = document.createElementNS(SVG_NS, 'path');
        el.id = id;
        el.setAttribute('fill', 'none');
        el.setAttribute('stroke', stroke);
        el.setAttribute('stroke-width', strokeWidth);
        if (dashArray) el.setAttribute('stroke-dasharray', dashArray);
        el.setAttribute('opacity', opacity);
        el.setAttribute('pointer-events', 'none');
        return el;
    }

    function makeLine(id, stroke, strokeWidth, dashArray) {
        const el = document.createElementNS(SVG_NS, 'line');
        el.id = id;
        el.setAttribute('stroke', stroke);
        el.setAttribute('stroke-width', strokeWidth);
        if (dashArray) el.setAttribute('stroke-dasharray', dashArray);
        el.setAttribute('pointer-events', 'none');
        return el;
    }

    // Spine curves
    g.appendChild(makePath('bz-spine-l', '#1A1A1A', '0.004', '0.015 0.008', '0.3'));
    g.appendChild(makePath('bz-spine-r', '#1A1A1A', '0.005', '0.015 0.008', '0.65'));

    // Control polygon lines: P0→P1 and P3→P2
    g.appendChild(makeLine('bz-ctrl-01', '#666', '0.003', '0.008 0.005'));
    g.appendChild(makeLine('bz-ctrl-32', '#666', '0.003', '0.008 0.005'));

    // Anchor P0 (non-draggable)
    const p0el = document.createElementNS(SVG_NS, 'circle');
    p0el.id = 'bz-p0';
    p0el.setAttribute('r', '0.012');
    p0el.setAttribute('fill', 'rgba(26,26,26,0.7)');
    p0el.setAttribute('stroke', 'white');
    p0el.setAttribute('stroke-width', '0.004');
    p0el.setAttribute('pointer-events', 'none');
    g.appendChild(p0el);

    // Draggable handles: P1 (CP1), P2 (CP2), P3 (tip)
    const handleDefs = [
        { id: 'bz-p1', geneX: 0, geneY: 1, color: 'rgba(90,180,175,0.7)', label: 'CP1' },
        { id: 'bz-p2', geneX: 2, geneY: 3, color: 'rgba(200,110,110,0.7)', label: 'CP2' },
        { id: 'bz-p3', geneX: 4, geneY: 5, color: 'rgba(195,155,80,0.7)',  label: 'Tip' },
    ];

    handleDefs.forEach(({ id, geneX, geneY, color }) => {
        const h = document.createElementNS(SVG_NS, 'circle');
        h.id = id;
        h.setAttribute('r', '0.013');
        h.setAttribute('fill', color);
        h.setAttribute('stroke', 'rgba(255,255,255,0.8)');
        h.setAttribute('stroke-width', '0.004');
        h.classList.add('bz-handle');
        h.dataset.geneX = geneX;
        h.dataset.geneY = geneY;
        h.addEventListener('mousedown', onBzHandleMouseDown);
        g.appendChild(h);
    });
}

function updateBezierOverlay() {
    const svg = document.getElementById('bezier-overlay');
    if (!svg || !svg.querySelector('#bz-group')) return;

    const pts = genomeToBezierPoints(currentGenome);
    const lpts = {
        p0: mirrorBzPoint(pts.p0),
        p1: mirrorBzPoint(pts.p1),
        p2: mirrorBzPoint(pts.p2),
        p3: mirrorBzPoint(pts.p3),
    };

    document.getElementById('bz-spine-r').setAttribute('d', bzPathD(pts.p0, pts.p1, pts.p2, pts.p3));
    document.getElementById('bz-spine-l').setAttribute('d', bzPathD(lpts.p0, lpts.p1, lpts.p2, lpts.p3));

    const ctrl01 = document.getElementById('bz-ctrl-01');
    ctrl01.setAttribute('x1', pts.p0.x); ctrl01.setAttribute('y1', pts.p0.y);
    ctrl01.setAttribute('x2', pts.p1.x); ctrl01.setAttribute('y2', pts.p1.y);

    const ctrl32 = document.getElementById('bz-ctrl-32');
    ctrl32.setAttribute('x1', pts.p3.x); ctrl32.setAttribute('y1', pts.p3.y);
    ctrl32.setAttribute('x2', pts.p2.x); ctrl32.setAttribute('y2', pts.p2.y);

    const p0el = document.getElementById('bz-p0');
    p0el.setAttribute('cx', pts.p0.x); p0el.setAttribute('cy', pts.p0.y);

    const p1el = document.getElementById('bz-p1');
    p1el.setAttribute('cx', pts.p1.x); p1el.setAttribute('cy', pts.p1.y);

    const p2el = document.getElementById('bz-p2');
    p2el.setAttribute('cx', pts.p2.x); p2el.setAttribute('cy', pts.p2.y);

    const p3el = document.getElementById('bz-p3');
    p3el.setAttribute('cx', pts.p3.x); p3el.setAttribute('cy', pts.p3.y);
}

function onBzHandleMouseDown(e) {
    e.preventDefault();
    const geneX = parseInt(e.target.dataset.geneX);
    const geneY = parseInt(e.target.dataset.geneY);

    bzDrag = { geneX, geneY, handleEl: e.target };
    e.target.classList.add('dragging');

    document.addEventListener('mousemove', onBzDragMove);
    document.addEventListener('mouseup', onBzDragEnd);
}

function onBzDragMove(e) {
    if (!bzDrag) return;

    const svg = document.getElementById('bezier-overlay');
    const pt = svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const svgPt = pt.matrixTransform(svg.getScreenCTM().inverse());

    // SVG viewBox coords → sim coords (flip y)
    const simX = svgPt.x;
    const simY = 1 - svgPt.y;

    const { geneX, geneY } = bzDrag;

    // Convert sim position back to genome values, then clamp to bounds
    const rawX = simX - BZ_START_X;
    const rawY = simY - BZ_START_Y;
    currentGenome[geneX] = Math.max(bounds.lower[geneX], Math.min(bounds.upper[geneX], rawX));
    currentGenome[geneY] = Math.max(bounds.lower[geneY], Math.min(bounds.upper[geneY], rawY));

    // Live update: overlay + sliders only, no full re-render
    updateBezierOverlay();
    updateSliders();
    updateGenomeDisplay();
}

function onBzDragEnd() {
    if (!bzDrag) return;

    bzDrag.handleEl.classList.remove('dragging');
    bzDrag = null;

    document.removeEventListener('mousemove', onBzDragMove);
    document.removeEventListener('mouseup', onBzDragEnd);

    // Full re-render only on release
    renderMorphology();
}

window.addEventListener('resize', () => {
    if (evoData.generations.length > 0) drawConvergenceChart();
});

// Start app
init();
