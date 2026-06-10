import { app } from "../../../scripts/app.js";

const PANEL_ID = "rwbt-director-panel";
const TOGGLE_ID = "rwbt-director-toggle";

function esc(text) {
  return String(text || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function chatContent(choice) {
  const content = choice?.message?.content;
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => (typeof part?.text === "string" ? part.text : ""))
      .filter(Boolean)
      .join("\n");
  }
  return "";
}

async function apiGet(path) {
  const response = await fetch(path);
  return response.json();
}

async function apiPost(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return response.json();
}

function ensureStyles() {
  if (document.getElementById("rwbt-director-styles")) return;
  const style = document.createElement("style");
  style.id = "rwbt-director-styles";
  style.textContent = `
    #${TOGGLE_ID} {
      position: fixed;
      right: 16px;
      bottom: 16px;
      z-index: 12000;
      border: 1px solid #466077;
      background: #15202c;
      color: #f6fbff;
      border-radius: 8px;
      padding: 8px 12px;
      font-size: 12px;
      cursor: pointer;
    }
    #${PANEL_ID} {
      position: fixed;
      right: 16px;
      bottom: 56px;
      width: 430px;
      max-height: 78vh;
      z-index: 12000;
      display: none;
      overflow: auto;
      border: 1px solid #394b5d;
      border-radius: 10px;
      background: #0f141a;
      color: #e8eef5;
      font: 12px/1.3 "Segoe UI", Tahoma, sans-serif;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
      padding: 10px;
    }
    #${PANEL_ID} h3 {
      margin: 0 0 6px;
      font-size: 13px;
      color: #9fd5ff;
    }
    #${PANEL_ID} .rwbt-row { margin: 8px 0; }
    #${PANEL_ID} textarea,
    #${PANEL_ID} input,
    #${PANEL_ID} select {
      width: 100%;
      box-sizing: border-box;
      border: 1px solid #33414f;
      background: #0a0e13;
      color: #e8eef5;
      border-radius: 6px;
      padding: 6px;
      font: inherit;
    }
    #${PANEL_ID} button {
      border: 1px solid #36526b;
      background: #153046;
      color: #f2f8ff;
      border-radius: 6px;
      padding: 5px 9px;
      margin-right: 6px;
      margin-top: 4px;
      cursor: pointer;
    }
    #${PANEL_ID} .rwbt-chat {
      height: 200px;
      overflow: auto;
      border: 1px solid #2f3e4c;
      border-radius: 6px;
      padding: 8px;
      background: #090d12;
    }
    #${PANEL_ID} .rwbt-msg { margin: 0 0 8px; }
    #${PANEL_ID} .rwbt-user { color: #8ed1ff; }
    #${PANEL_ID} .rwbt-assistant { color: #b3f0c6; }
    #${PANEL_ID} .rwbt-meta { color: #9ba8b5; font-size: 11px; }
    #${PANEL_ID} .rwbt-jobs {
      max-height: 130px;
      overflow: auto;
      border: 1px solid #2f3e4c;
      border-radius: 6px;
      padding: 6px;
      background: #090d12;
    }
    #${PANEL_ID} .rwbt-link {
      display: block;
      color: #86d2ff;
      text-decoration: none;
      margin-bottom: 2px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    #${PANEL_ID} .rwbt-thumb-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(88px, 1fr));
      gap: 8px;
      margin-top: 8px;
    }
    #${PANEL_ID} .rwbt-thumb {
      width: 100%;
      height: 88px;
      object-fit: cover;
      border: 1px solid #2f3e4c;
      border-radius: 6px;
      background: #0b1015;
      cursor: pointer;
    }
  `;
  document.head.appendChild(style);
}

function buildPanel() {
  if (document.getElementById(PANEL_ID)) return;

  ensureStyles();

  const panel = document.createElement("div");
  panel.id = PANEL_ID;
  panel.innerHTML = `
    <h3>RWBT Director</h3>
    <div class="rwbt-row rwbt-meta" id="rwbt-status">Status: checking...</div>
    <div class="rwbt-row">
      <label>Session</label>
      <input id="rwbt-session" value="rwbt-main" />
    </div>
    <div class="rwbt-row">
      <label>Active Plan</label>
      <textarea id="rwbt-plan" rows="4" placeholder="Director plan remains active until replaced."></textarea>
      <input id="rwbt-plan-file" type="file" accept=".md,.txt,text/plain" style="display:none" />
      <button id="rwbt-load-plan">Load Plan</button>
      <button id="rwbt-set-plan">Set Plan</button>
      <button id="rwbt-upload-plan">Upload Plan File</button>
      <button id="rwbt-clear-plan">Clear Plan</button>
      <div id="rwbt-plan-file-meta" class="rwbt-meta"></div>
    </div>
    <div class="rwbt-row">
      <label>Chat</label>
      <div class="rwbt-chat" id="rwbt-chat"></div>
      <textarea id="rwbt-input" rows="3" placeholder="Ask director..." ></textarea>
      <button id="rwbt-send">Send</button>
      <button id="rwbt-clear-chat">Clear Local Chat</button>
    </div>
    <div class="rwbt-row">
      <label>RWBT Jobs</label>
      <div class="rwbt-jobs" id="rwbt-jobs"></div>
      <button id="rwbt-refresh-jobs">Refresh Jobs</button>
    </div>
    <div class="rwbt-row">
      <label>Latest Outputs</label>
      <div class="rwbt-jobs" id="rwbt-outputs"></div>
    </div>
  `;

  const toggle = document.createElement("button");
  toggle.id = TOGGLE_ID;
  toggle.textContent = "RWBT Director";
  toggle.addEventListener("click", () => {
    panel.style.display = panel.style.display === "none" || !panel.style.display ? "block" : "none";
  });

  document.body.appendChild(panel);
  document.body.appendChild(toggle);

  const $ = (id) => panel.querySelector(`#${id}`);
  const state = {
    chat: [],
    selectedJobId: "",
    selectedOutputRoot: "",
  };

  const renderChat = () => {
    const html = state.chat
      .map((msg) => {
        const cls = msg.role === "assistant" ? "rwbt-assistant" : "rwbt-user";
        return `<div class="rwbt-msg ${cls}"><b>${esc(msg.role)}:</b> ${esc(msg.content)}</div>`;
      })
      .join("");
    $("rwbt-chat").innerHTML = html;
    $("rwbt-chat").scrollTop = $("rwbt-chat").scrollHeight;
  };

  const sessionId = () => ($("rwbt-session").value || "rwbt-main").trim() || "rwbt-main";

  const updateStatus = async () => {
    const health = await apiGet("/rwbt_director_ui/health");
    if (health && !health.error) {
      $("rwbt-status").textContent = `Status: online (${health.status || "ok"})`;
    } else {
      $("rwbt-status").textContent = `Status: offline (${health?.error || "unknown"})`;
    }
  };

  const loadPlan = async () => {
    const plan = await apiGet(`/rwbt_director_ui/plan?session_id=${encodeURIComponent(sessionId())}`);
    if (plan?.plan?.plan_text) {
      $("rwbt-plan").value = plan.plan.plan_text;
    } else {
      $("rwbt-plan").value = "";
    }
  };

  const setPlan = async () => {
    const result = await apiPost("/rwbt_director_ui/plan", {
      session_id: sessionId(),
      plan_text: $("rwbt-plan").value || "",
    });
    if (result?.error) {
      alert(`Set plan failed: ${result.error}`);
    }
  };

  const clearPlan = async () => {
    await apiPost("/rwbt_director_ui/clear_plan", { session_id: sessionId() });
    $("rwbt-plan").value = "";
    $("rwbt-plan-file-meta").textContent = "";
  };

  const uploadPlanFromLocal = async (file) => {
    if (!file) return;
    const text = await file.text();
    $("rwbt-plan").value = text || "";
    $("rwbt-plan-file-meta").textContent = `Loaded: ${file.name} (${Math.max(0, text.length)} chars)`;
    const result = await apiPost("/rwbt_director_ui/plan", {
      session_id: sessionId(),
      plan_text: text || "",
      plan_id: file.name || "uploaded-plan",
    });
    if (result?.error) {
      alert(`Upload plan failed: ${result.error}`);
      return;
    }
    $("rwbt-plan-file-meta").textContent = `Loaded + Applied: ${file.name} (${Math.max(0, text.length)} chars)`;
  };

  const sendChat = async () => {
    const input = $("rwbt-input");
    const text = (input.value || "").trim();
    if (!text) return;
    input.value = "";

    state.chat.push({ role: "user", content: text });
    renderChat();

    const result = await apiPost("/rwbt_director_ui/chat", {
      session_id: sessionId(),
      user_message: text,
      persist_context: true,
      max_tokens: 1200,
      temperature: 0.2,
      model: "",
    });

    if (result?.error) {
      state.chat.push({ role: "assistant", content: `Error: ${result.error}` });
      renderChat();
      return;
    }

    const assistant = chatContent(result?.choices?.[0]) || "(no assistant content)";
    state.chat.push({ role: "assistant", content: assistant });
    renderChat();
  };

  const refreshJobs = async () => {
    const data = await apiGet("/rwbt_director_ui/jobs");
    const jobs = Array.isArray(data?.jobs) ? data.jobs : [];
    $("rwbt-jobs").innerHTML = jobs
      .map((job) => {
        const id = esc(job.job_id);
        const status = esc(job.status || "unknown");
        return `<a class="rwbt-link" data-job="${id}" href="#">${id} [${status}] c:${job.completed || 0} f:${job.failed || 0}</a>`;
      })
      .join("");

    panel.querySelectorAll("[data-job]").forEach((el) => {
      el.addEventListener("click", async (event) => {
        event.preventDefault();
        state.selectedJobId = el.getAttribute("data-job") || "";
        await refreshOutputs();
      });
    });
  };

  const refreshOutputs = async () => {
    if (!state.selectedJobId) {
      $("rwbt-outputs").innerHTML = "<div class='rwbt-meta'>Select a job above.</div>";
      return;
    }
    const data = await apiGet(`/rwbt_director_ui/job_outputs?job_id=${encodeURIComponent(state.selectedJobId)}&limit=30`);
    const images = Array.isArray(data?.images) ? data.images : [];
    state.selectedOutputRoot = data?.job_dir ? String(data.job_dir).replace(/[\\/]?[^\\/]+$/, "") : "";
    const imageThumbs = images
      .map((path) => {
        const src = `/rwbt_director_ui/image?path=${encodeURIComponent(path)}`;
        return `<img class="rwbt-thumb" src="${src}" data-path="${esc(path)}" title="${esc(path)}" />`;
      })
      .join("");
    const statePath = esc(data?.state_path || "");
    const manifestPath = esc(data?.manifest_path || "");

    $("rwbt-outputs").innerHTML = `
      <div class="rwbt-meta">Job: ${esc(state.selectedJobId)}</div>
      <div class="rwbt-link">${statePath}</div>
      <div class="rwbt-link">${manifestPath}</div>
      ${imageThumbs ? `<div class="rwbt-thumb-grid">${imageThumbs}</div>` : "<div class='rwbt-meta'>No images found.</div>"}
    `;

    panel.querySelectorAll(".rwbt-thumb").forEach((el) => {
      el.addEventListener("click", () => {
        const path = el.getAttribute("data-path") || "";
        const src = `/rwbt_director_ui/image?path=${encodeURIComponent(path)}`;
        window.open(src, "_blank", "noopener,noreferrer");
      });
    });
  };

  $("rwbt-load-plan").addEventListener("click", loadPlan);
  $("rwbt-set-plan").addEventListener("click", setPlan);
  $("rwbt-upload-plan").addEventListener("click", () => $("rwbt-plan-file").click());
  $("rwbt-plan-file").addEventListener("change", async (event) => {
    const file = event?.target?.files?.[0];
    await uploadPlanFromLocal(file);
    event.target.value = "";
  });
  $("rwbt-clear-plan").addEventListener("click", clearPlan);
  $("rwbt-send").addEventListener("click", sendChat);
  $("rwbt-clear-chat").addEventListener("click", () => {
    state.chat = [];
    renderChat();
  });
  $("rwbt-refresh-jobs").addEventListener("click", refreshJobs);

  updateStatus();
  loadPlan();
  refreshJobs();
  refreshOutputs();
}

app.registerExtension({
  name: "rwbt.director.panel",
  setup() {
    buildPanel();
  },
});
