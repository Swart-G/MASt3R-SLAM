"use strict";
const $ = (id) => document.getElementById(id);
let token = "",
  items = [],
  state = {},
  lastQr = "",
  pending = false;
const phases = {
  IDLE: "Не запущен",
  LOADING: "Подготовка",
  READY: "Готов",
  SCANNING: "Сканирование",
  PAUSED: "На паузе",
  AR_PREPARING: "Подготовка AR",
  AR_VIEW: "Просмотр AR",
  FINALIZING: "Сохранение",
  SAVED: "Сохранён",
  DISCONNECTED: "Связь потеряна",
  ERROR: "Ошибка",
};
function notice(message = "") {
  $("notice").textContent = message;
  $("notice").hidden = !message;
}
async function api(path, options = {}) {
  const response = await fetch("/api/" + path, {
    ...options,
    headers: { "Content-Type": "application/json", "X-Panel-Token": token },
  });
  if (!response.ok) {
    const body = await response.text();
    let message = body;
    try {
      message = JSON.parse(body).error || body;
    } catch {}
    throw new Error(message);
  }
  return response.json();
}
function route() {
  const projects = location.hash === "#projects";
  $("scan-page").hidden = projects;
  $("projects-page").hidden = !projects;
  $("nav-scan").classList.toggle("active", !projects);
  $("nav-projects").classList.toggle("active", projects);
  if (projects) loadProjects().catch((error) => notice(error.message));
}
window.addEventListener("hashchange", route);
async function confirmAction(title, copy) {
  $("confirm-title").textContent = title;
  $("confirm-copy").textContent = copy;
  const dialog = $("confirm-dialog");
  dialog.showModal();
  return new Promise((resolve) =>
    dialog.addEventListener(
      "close",
      () => resolve(dialog.returnValue === "confirm"),
      { once: true },
    ),
  );
}
async function action(callback) {
  if (pending) return;
  pending = true;
  try {
    notice();
    await callback();
    await poll();
  } catch (error) {
    notice(error.message);
  } finally {
    pending = false;
  }
}
async function newSession(project_id = null, name = null) {
  await api("session", {
    method: "POST",
    body: JSON.stringify({ project_id, name }),
  });
  location.hash = "#scan";
}
$("new-session-form").onsubmit = (event) => {
  event.preventDefault();
  const input = $("project-name");
  input.setCustomValidity(input.value.trim() ? "" : "Введите название проекта");
  if (input.reportValidity()) $("new-session-dialog").close("create");
};
$("project-name").oninput = () => $("project-name").setCustomValidity("");
$("cancel-session").onclick = () => $("new-session-dialog").close("cancel");
$("new-session").onclick = async () => {
  const dialog = $("new-session-dialog");
  dialog.returnValue = "cancel";
  $("new-session-form").reset();
  $("project-name").setCustomValidity("");
  dialog.showModal();
  const confirmed = await new Promise((resolve) =>
    dialog.addEventListener(
      "close",
      () => resolve(dialog.returnValue === "create"),
      { once: true },
    ),
  );
  if (confirmed)
    await action(() => newSession(null, $("project-name").value.trim()));
};
for (const command of ["pause", "resume", "stop"]) {
  $(command).onclick = () =>
    action(async () => {
      if (
        command === "stop" &&
        !(await confirmAction(
          "Завершить сканирование?",
          "Текущая реконструкция будет сохранена в новой ревизии проекта.",
        ))
      )
        return;
      await api("commands/" + command, { method: "POST", body: "{}" });
      if (command === "stop") await loadProjects();
    });
}
function cell(row, text, tag = "td") {
  const element = document.createElement(tag);
  element.textContent = text;
  row.append(element);
  return element;
}
function renderProjects() {
  const query = $("search").value.toLowerCase();
  const filtered = items.filter((item) =>
    (item.name + " " + item.project_id).toLowerCase().includes(query),
  );
  $("project-rows").replaceChildren();
  $("empty-projects").hidden = filtered.length > 0;
  for (const item of filtered) {
    const row = document.createElement("tr");
    const name = cell(row, item.name);
    const id = document.createElement("small");
    id.textContent = item.project_id;
    name.append(id);
    cell(row, new Date(item.updated_at).toLocaleString("ru-RU"));
    cell(row, item.keyframes);
    cell(row, (item.size_bytes / 1048576).toFixed(1) + " МБ");
    cell(row, String(item.revision).padStart(2, "0"));
    const actions = cell(row, "");
    const group = document.createElement("div");
    group.className = "d-flex gap-2";
    actions.append(group);
    const open = document.createElement("button");
    open.className = "btn btn-outline-light btn-sm";
    open.textContent = "Продолжить";
    open.disabled =
      !item.revision ||
      state.busy ||
      !["IDLE", "READY", "SAVED", "ERROR"].includes(state.phase);
    open.onclick = () => action(() => newSession(item.project_id));
    group.append(open);
    const link = document.createElement("a");
    link.className = "btn btn-outline-light btn-sm";
    link.textContent = "↓ PLY";
    link.href =
      "/api/projects/" + encodeURIComponent(item.project_id) + "/export";
    group.append(link);
    const remove = document.createElement("button");
    remove.className = "btn btn-outline-danger btn-sm";
    remove.textContent = "Удалить";
    remove.disabled = state.project_id === item.project_id;
    remove.onclick = () =>
      action(async () => {
        if (
          await confirmAction(
            "Удалить проект?",
            `«${item.name}» и все его ревизии будут удалены без возможности восстановления.`,
          )
        ) {
          await api("projects/" + encodeURIComponent(item.project_id), {
            method: "DELETE",
          });
          await loadProjects();
        }
      });
    group.append(remove);
    $("project-rows").append(row);
  }
}
async function loadProjects() {
  items = await api("projects");
  $("project-count").textContent = items.length;
  renderProjects();
}
$("refresh-projects").onclick = () => action(loadProjects);
$("search").oninput = renderProjects;
async function poll() {
  try {
    state = await api("status");
    token = state.csrf;
    const title = state.project_name || "Новый проект";
    $("project-title").textContent = title;
    document.title = `${title} — MASt3R-SLAM`;
    $("session-state").textContent = phases[state.phase] || state.phase;
    $("session-state").hidden = ["IDLE", "READY"].includes(state.phase);
    $("phone-help").textContent = state.connected
      ? "Запустите сканирование в приложении."
      : "Отсканируйте QR-код в Android-приложении.";
    $("new-session").disabled =
      state.busy || !["IDLE", "READY", "SAVED", "ERROR"].includes(state.phase);
    $("pause").disabled = state.busy || state.phase !== "SCANNING";
    $("resume").disabled =
      state.busy ||
      !state.connected ||
      !["PAUSED", "AR_VIEW"].includes(state.phase);
    $("stop").disabled =
      state.busy ||
      !["SCANNING", "PAUSED", "AR_VIEW", "DISCONNECTED"].includes(
        state.phase,
      ) ||
      !state.project_id;
    $("pair-placeholder").hidden = !!state.pairing_uri;
    $("pair-content").hidden = !state.pairing_uri;
    if (state.pairing_uri && lastQr !== state.pairing_uri) {
      $("qr").src = "/api/pairing.png?t=" + Date.now();
      $("pair-uri").textContent = state.pairing_uri;
      lastQr = state.pairing_uri;
    }
    if (!state.pairing_uri) lastQr = "";
    $("scene-empty").hidden = state.preview;
    $("preview").hidden = !state.preview;
    if (state.preview) $("preview").src = "/api/preview.jpg?t=" + Date.now();
    $("scene-hint").textContent =
      state.phase === "LOADING"
        ? "Подготовка сеанса…"
        : state.busy
          ? "Выполняется операция…"
          : state.stats.guidance || "";
    $("scene-hint").hidden = !$("scene-hint").textContent;
    if (state.error) notice(state.error);
    if (state.events.length) {
      $("events").replaceChildren();
      for (const event of state.events.slice(-4).reverse()) {
        const row = document.createElement("div");
        row.className = "event-row";
        const label = document.createElement("span");
        label.textContent =
          {
            state_changed: "Состояние",
            map_updated: "Карта",
            progress: "Операция",
            save_completed: "Сохранено",
            finalized: "Завершено",
            error: "Ошибка",
            camera_mismatch: "Камера",
            capacity_warning: "Память",
            tracking_lost: "Трекинг",
          }[event.name] || event.name;
        const copy = document.createElement("p");
        copy.textContent = phases[event.detail] || event.detail;
        row.append(label, copy);
        $("events").append(row);
      }
    } else {
      $("events").textContent = "Событий пока нет.";
    }
  } catch (error) {
    throw error;
  }
}
(async () => {
  route();
  try {
    await poll();
    await loadProjects();
  } catch (error) {
    notice(error.message);
  } finally {
    setInterval(() => poll().catch((error) => notice(error.message)), 1500);
  }
})();
