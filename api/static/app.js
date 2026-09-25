const engineSelect = document.getElementById("engine-select");
const playBtn = document.getElementById("play-btn");
const resetBtn = document.getElementById("reset-btn");
const cycleLabel = document.getElementById("cycle-label");
const rulValueEl = document.getElementById("rul-value");
const riskBadgeEl = document.getElementById("risk-badge");
const canvas = document.getElementById("rul-chart");
const ctx = canvas.getContext("2d");

const WINDOW = 5;      // must match the model's training window
const STEP_MS = 400;   // simulation speed

let currentHistory = null;
let simTimer = null;
let simIndex = 0;
let plottedPoints = [];

async function loadEngines() {
    const res = await fetch("/engines");
    const data = await res.json();
    engineSelect.innerHTML = "";
    for (const eng of data.engines) {
        const opt = document.createElement("option");
        opt.value = eng.engine_id;
        opt.textContent = `Engine ${eng.engine_id} (${eng.total_cycles} cycles to failure)`;
        engineSelect.appendChild(opt);
    }
}

async function loadHistory(engineId) {
    const res = await fetch(`/engines/${engineId}/history`);
    return res.json();
}

function resetSimState() {
    clearInterval(simTimer);
    simTimer = null;
    simIndex = 0;
    plottedPoints = [];
    rulValueEl.textContent = "--";
    setRiskBadge("healthy", "HEALTHY");
    cycleLabel.textContent = "Cycle: -";
    drawChart();
}

function setRiskBadge(band, label) {
    riskBadgeEl.className = `risk-badge ${band}`;
    riskBadgeEl.textContent = label;
}

async function stepSimulation() {
    if (!currentHistory) return;
    const cycles = currentHistory.cycles;
    if (simIndex >= cycles.length) {
        clearInterval(simTimer);
        simTimer = null;
        playBtn.textContent = "Start Simulation";
        return;
    }

    const windowStart = Math.max(0, simIndex - WINDOW + 1);
    const windowSlice = cycles.slice(windowStart, simIndex + 1);
    const readings = windowSlice.map(({ cycle, ...sensors }) => sensors);

    const payload = { engine_id: currentHistory.engine_id, readings };
    const res = await fetch("/predict/rul", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });
    const pred = await res.json();

    rulValueEl.textContent = pred.predicted_rul;
    setRiskBadge(pred.risk_band, pred.risk_band.toUpperCase());
    cycleLabel.textContent = `Cycle: ${cycles[simIndex].cycle} / ${cycles[cycles.length - 1].cycle}`;

    plottedPoints.push({ cycle: cycles[simIndex].cycle, rul: pred.predicted_rul, band: pred.risk_band });
    drawChart();

    simIndex += 1;
}

function drawChart() {
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const padding = 40;
    const maxRul = 130;

    const zones = [
        { upTo: 20, color: "rgba(239, 68, 68, 0.12)" },
        { upTo: 50, color: "rgba(245, 185, 66, 0.10)" },
        { upTo: maxRul, color: "rgba(46, 204, 113, 0.07)" },
    ];
    let prev = 0;
    for (const z of zones) {
        const yTop = h - padding - (z.upTo / maxRul) * (h - 2 * padding);
        const yBottom = h - padding - (prev / maxRul) * (h - 2 * padding);
        ctx.fillStyle = z.color;
        ctx.fillRect(padding, yTop, w - 2 * padding, yBottom - yTop);
        prev = z.upTo;
    }

    if (plottedPoints.length === 0) return;

    const maxCycle = Math.max(...plottedPoints.map((p) => p.cycle), 10);
    const xFor = (cycle) => padding + (cycle / maxCycle) * (w - 2 * padding);
    const yFor = (rul) => h - padding - (Math.min(rul, maxRul) / maxRul) * (h - 2 * padding);

    ctx.strokeStyle = "#4f8cff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    plottedPoints.forEach((p, i) => {
        const x = xFor(p.cycle), y = yFor(p.rul);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();

    const bandColor = { healthy: "#2ecc71", warning: "#f5b942", critical: "#ef4444" };
    for (const p of plottedPoints) {
        ctx.fillStyle = bandColor[p.band] || "#4f8cff";
        ctx.beginPath();
        ctx.arc(xFor(p.cycle), yFor(p.rul), 3, 0, Math.PI * 2);
        ctx.fill();
    }

    ctx.fillStyle = "#8b95ab";
    ctx.font = "12px sans-serif";
    ctx.fillText("Predicted RUL (cycles)", padding, 18);
    ctx.fillText(`Cycle ${maxCycle}`, w - padding - 60, h - 12);
}

playBtn.addEventListener("click", async () => {
    if (simTimer) {
        clearInterval(simTimer);
        simTimer = null;
        playBtn.textContent = "Resume Simulation";
        return;
    }
    if (!currentHistory) {
        currentHistory = await loadHistory(engineSelect.value);
    }
    playBtn.textContent = "Pause";
    simTimer = setInterval(stepSimulation, STEP_MS);
});

resetBtn.addEventListener("click", async () => {
    currentHistory = await loadHistory(engineSelect.value);
    playBtn.textContent = "Start Simulation";
    resetSimState();
});

engineSelect.addEventListener("change", async () => {
    currentHistory = await loadHistory(engineSelect.value);
    playBtn.textContent = "Start Simulation";
    resetSimState();
});

(async function init() {
    await loadEngines();
    currentHistory = await loadHistory(engineSelect.value);
    resetSimState();
})();