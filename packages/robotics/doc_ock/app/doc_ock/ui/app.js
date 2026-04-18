const API_BASE = "";

const $ = (id) => document.getElementById(id);

const statusDot = $("statusDot");
const statusText = $("statusText");
const log = $("log");
const voiceToggle = $("voiceToggle");
const voiceLabel = $("voiceLabel");

let lastSessionState = null;
let pollTimer = null;

function ts() {
  const d = new Date();
  return d.toTimeString().slice(0, 8);
}

function logLine(level, msg) {
  const line = document.createElement("div");
  line.innerHTML = `<span class="time">[${ts()}]</span> <span class="${level}">${msg}</span>`;
  log.appendChild(line);
  log.scrollTop = log.scrollHeight;
}

async function apiFetch(path, options = {}) {
  try {
    const res = await fetch(API_BASE + path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    const text = await res.text();
    let body = null;
    try { body = text ? JSON.parse(text) : null; } catch { body = text; }
    if (!res.ok) {
      const msg = body && body.detail ? body.detail : `HTTP ${res.status}`;
      logLine("err", `${options.method || "GET"} ${path} -> ${res.status}: ${msg}`);
      return { ok: false, status: res.status, body };
    }
    return { ok: true, status: res.status, body };
  } catch (e) {
    logLine("err", `${options.method || "GET"} ${path} -> network error: ${e.message}`);
    return { ok: false, status: 0, body: null };
  }
}

function setConnection(ok) {
  if (ok) {
    statusDot.className = "dot ok";
    statusText.textContent = "connected";
  } else {
    statusDot.className = "dot err";
    statusText.textContent = "offline";
  }
}

function renderHealth(data) {
  $("healthState").textContent = data.state;
  $("healthDry").textContent = data.dry_run ? "yes" : "no";
  $("healthModel").textContent = data.model_loaded ? "yes" : "no";
  $("healthVoice").textContent = data.voice_mode_enabled ? "ON" : "OFF";
  $("healthCams").textContent = JSON.stringify(data.configured_cameras, null, 2);
  applyVoice(data.voice_mode_enabled);
  renderAudio({
    enabled: data.audio_enabled,
    running: data.audio_running,
    latest_partial: data.latest_partial_transcript,
    latest_committed: data.latest_committed_transcript,
    error: data.audio_error,
  });
}

function renderAudio(a) {
  const pill = $("audioPill");
  if (!a.enabled) {
    pill.textContent = "audio off";
    pill.className = "pill";
  } else if (a.running) {
    pill.textContent = "listening";
    pill.className = "pill running";
  } else {
    pill.textContent = "audio idle";
    pill.className = "pill stopped";
  }
  $("partialText").textContent = a.latest_partial || "-";
  $("committedText").textContent = a.latest_committed || "-";
  $("audioErr").textContent = a.error || "-";
}

function setSessionPill(state) {
  const pill = $("sessionPill");
  pill.className = "pill " + (state || "");
  pill.textContent = state || "-";
  if (state === "running") {
    statusDot.className = "dot running";
    statusText.textContent = "session running";
  }
}

function renderSession(data) {
  $("sState").textContent = data.state;
  $("sStep").textContent = data.step_count;
  $("sTask").textContent = data.task || "-";
  $("sModel").textContent = data.model_repo_id || "-";
  $("sStarted").textContent = data.started_at || "-";
  $("sVoice").textContent = data.voice_mode_enabled ? "ON" : "OFF";
  $("sErr").textContent = data.last_error || "-";
  setSessionPill(data.state);

  if (typeof data.latest_partial_transcript !== "undefined") {
    $("partialText").textContent = data.latest_partial_transcript || "-";
  }
  if (typeof data.latest_committed_transcript !== "undefined") {
    $("committedText").textContent = data.latest_committed_transcript || "-";
  }
  if (typeof data.audio_error !== "undefined") {
    $("audioErr").textContent = data.audio_error || "-";
  }

  if (data.state !== lastSessionState) {
    logLine("info", `session state: ${lastSessionState || "-"} -> ${data.state}`);
    lastSessionState = data.state;
  }
}

function applyVoice(enabled) {
  voiceToggle.checked = !!enabled;
  voiceLabel.textContent = enabled ? "ON" : "OFF";
  voiceLabel.classList.toggle("on", !!enabled);
}

async function fetchHealth() {
  const r = await apiFetch("/health");
  if (r.ok) {
    setConnection(true);
    renderHealth(r.body);
  } else {
    setConnection(false);
  }
}

async function fetchSession() {
  const r = await apiFetch("/session/status");
  if (r.ok) renderSession(r.body);
}

async function fetchVoice() {
  const r = await apiFetch("/voice-mode");
  if (r.ok) applyVoice(r.body.enabled);
}

async function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  const tick = async () => {
    await fetchHealth();
    await fetchSession();
  };
  await tick();
  pollTimer = setInterval(tick, 1000);
}

$("sessionForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const body = {
    task: $("task").value.trim(),
    model_repo_id: $("modelRepoId").value.trim(),
  };
  const steps = parseInt($("maxSteps").value, 10);
  const duration = parseFloat($("maxDuration").value);
  if (!Number.isNaN(steps) && steps > 0) body.max_steps = steps;
  if (!Number.isNaN(duration) && duration > 0) body.max_duration_s = duration;

  logLine("info", `starting session: ${body.task} (${body.model_repo_id})`);
  const r = await apiFetch("/session/start", {
    method: "POST",
    body: JSON.stringify(body),
  });
  if (r.ok) {
    logLine("ok", `session started: state=${r.body.state}`);
    renderSession(r.body);
  }
});

$("stopBtn").addEventListener("click", async () => {
  logLine("info", "stopping session");
  const r = await apiFetch("/session/stop", { method: "POST" });
  if (r.ok) {
    logLine("ok", `stop acknowledged: state=${r.body.state}`);
    renderSession(r.body);
  }
});

voiceToggle.addEventListener("change", async () => {
  const desired = voiceToggle.checked;
  const r = await apiFetch("/voice-mode", {
    method: "POST",
    body: JSON.stringify({ enabled: desired }),
  });
  if (r.ok) {
    applyVoice(r.body.enabled);
    logLine("ok", `voice mode -> ${r.body.enabled ? "ON" : "OFF"}`);
  } else {
    applyVoice(!desired);
  }
});

$("refreshHealthBtn").addEventListener("click", fetchHealth);
$("clearLogBtn").addEventListener("click", () => { log.innerHTML = ""; });

document.addEventListener("keydown", (e) => {
  const tag = (e.target && e.target.tagName) || "";
  if (tag === "INPUT" || tag === "TEXTAREA") return;
  if (e.key === "v" || e.key === "V") {
    voiceToggle.checked = !voiceToggle.checked;
    voiceToggle.dispatchEvent(new Event("change"));
  }
});

logLine("info", "UI ready");
startPolling();
