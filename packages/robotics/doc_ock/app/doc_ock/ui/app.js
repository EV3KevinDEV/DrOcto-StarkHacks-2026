const API_BASE = "";

const $ = (id) => document.getElementById(id);

const statusDot = $("statusDot");
const statusText = $("statusText");
const log = $("log");
const voiceToggle = $("voiceToggle");
const voiceLabel = $("voiceLabel");
const commitTranscriptBtn = $("commitTranscriptBtn");
const voiceActionBtn = $("voiceActionBtn");

let lastSessionState = null;
let pollTimer = null;
let voiceActionWorkflowRunning = false;

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
  $("healthRobot").textContent = `${data.robot_type} @ ${data.robot_port} (${data.robot_id})`;
  $("healthTeleop").textContent = data.teleop_port
    ? `${data.teleop_type} @ ${data.teleop_port} (${data.teleop_id})`
    : `${data.teleop_type} (${data.teleop_id})`;
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

function getSessionBodyFromForm() {
  const body = {
    task: $("task").value.trim(),
    model_repo_id: $("modelRepoId").value.trim(),
  };
  const steps = parseInt($("maxSteps").value, 10);
  const duration = parseFloat($("maxDuration").value);
  if (!Number.isNaN(steps) && steps > 0) body.max_steps = steps;
  if (!Number.isNaN(duration) && duration > 0) body.max_duration_s = duration;
  return body;
}

function transcriptValue(text) {
  const normalized = (text || "").trim();
  return normalized === "-" ? "" : normalized;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isSessionActive(state) {
  return ["starting", "running", "stopping"].includes((state || "").trim().toLowerCase());
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
  commitTranscriptBtn.disabled = !enabled;
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

async function commitTranscript() {
  const r = await apiFetch("/audio/commit", { method: "POST" });
  if (r.ok) {
    if (typeof r.body.latest_partial_transcript !== "undefined") {
      $("partialText").textContent = r.body.latest_partial_transcript || "-";
    }
    if (typeof r.body.latest_committed_transcript !== "undefined") {
      $("committedText").textContent = r.body.latest_committed_transcript || "-";
    }
    if (typeof r.body.audio_error !== "undefined") {
      $("audioErr").textContent = r.body.audio_error || "-";
    }
    logLine("ok", "manual transcript commit requested");
  }
  return r;
}

async function setVoiceMode(enabled) {
  const r = await apiFetch("/voice-mode", {
    method: "POST",
    body: JSON.stringify({ enabled }),
  });
  if (r.ok) {
    applyVoice(r.body.enabled);
    logLine("ok", `voice mode -> ${r.body.enabled ? "ON" : "OFF"}`);
  } else {
    applyVoice(!enabled);
  }
  return r;
}

async function startSessionFromForm() {
  const body = getSessionBodyFromForm();
  logLine("info", `starting session: ${body.task} (${body.model_repo_id})`);
  const r = await apiFetch("/session/start", {
    method: "POST",
    body: JSON.stringify(body),
  });
  if (r.ok) {
    logLine("ok", `session started: state=${r.body.state}`);
    renderSession(r.body);
  }
  return r;
}

async function waitForCommittedTranscriptChange(previousCommitted, timeoutMs = 5000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const r = await apiFetch("/health");
    if (!r.ok) return "";
    renderHealth(r.body);
    const latestCommitted = transcriptValue(r.body.latest_committed_transcript);
    if (latestCommitted && latestCommitted !== previousCommitted) return latestCommitted;
    await sleep(250);
  }
  return "";
}

async function waitForSessionToSettle(timeoutMs = 5000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const r = await apiFetch("/session/status");
    if (!r.ok) return null;
    renderSession(r.body);
    if (!isSessionActive(r.body.state)) return r.body;
    await sleep(250);
  }
  return null;
}

async function stopSessionIfActive() {
  const currentState = ($("sState").textContent || "").trim().toLowerCase();
  if (!isSessionActive(currentState)) return true;

  logLine("info", "interrupting current session for voice-action");
  const r = await apiFetch("/session/stop", { method: "POST" });
  if (!r.ok) return false;
  renderSession(r.body);

  const settled = await waitForSessionToSettle();
  if (!settled) {
    logLine("err", "timed out waiting for the current session to stop");
    return false;
  }

  return true;
}

async function runVoiceActionWorkflow() {
  if (voiceActionWorkflowRunning) return;

  voiceActionWorkflowRunning = true;
  const originalLabel = voiceActionBtn.textContent;
  voiceActionBtn.disabled = true;
  voiceActionBtn.textContent = "Listening...";

  try {
    const stopped = await stopSessionIfActive();
    if (!stopped) return;

    const voiceResult = await setVoiceMode(true);
    if (!voiceResult.ok) return;

    logLine("info", "speak now; the UI will auto-commit after a short pause");

    const initialCommitted = transcriptValue($("committedText").textContent);
    let lastPartial = transcriptValue($("partialText").textContent);
    let sawSpeech = false;
    let lastChangeAt = Date.now();
    const deadline = Date.now() + 15000;

    while (Date.now() < deadline) {
      const health = await apiFetch("/health");
      if (!health.ok) return;
      renderHealth(health.body);

      const partial = transcriptValue(health.body.latest_partial_transcript);
      const latestCommitted = transcriptValue(health.body.latest_committed_transcript);

      if (latestCommitted && latestCommitted !== initialCommitted) {
        $("task").value = latestCommitted;
        await setVoiceMode(false);
        logLine("ok", `voice command committed: ${latestCommitted}`);
        await startSessionFromForm();
        return;
      }

      if (partial && partial !== lastPartial) {
        sawSpeech = true;
        lastPartial = partial;
        lastChangeAt = Date.now();
      }

      if (sawSpeech && Date.now() - lastChangeAt >= 1200) {
        const commitResult = await commitTranscript();
        if (!commitResult.ok) return;

        const committed = await waitForCommittedTranscriptChange(initialCommitted);
        if (committed) {
          $("task").value = committed;
          await setVoiceMode(false);
          logLine("ok", `voice command committed: ${committed}`);
          await startSessionFromForm();
          return;
        }

        logLine("err", "voice commit was requested, but no committed transcript arrived");
        return;
      }

      await sleep(300);
    }

    logLine("err", "voice-action test timed out waiting for speech");
  } finally {
    voiceActionBtn.textContent = originalLabel;
    voiceActionBtn.disabled = false;
    voiceActionWorkflowRunning = false;
  }
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
  await startSessionFromForm();
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
  await setVoiceMode(voiceToggle.checked);
});

commitTranscriptBtn.addEventListener("click", commitTranscript);
voiceActionBtn.addEventListener("click", runVoiceActionWorkflow);

$("refreshHealthBtn").addEventListener("click", fetchHealth);
$("clearLogBtn").addEventListener("click", () => { log.innerHTML = ""; });

document.addEventListener("keydown", (e) => {
  const tag = (e.target && e.target.tagName) || "";
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "BUTTON" || tag === "SELECT") return;
  if (e.key === "v" || e.key === "V") {
    voiceToggle.checked = !voiceToggle.checked;
    voiceToggle.dispatchEvent(new Event("change"));
    return;
  }
  if (e.key === "Enter") {
    e.preventDefault();
    runVoiceActionWorkflow();
  }
});

logLine("info", "UI ready");
commitTranscriptBtn.disabled = true;
startPolling();
