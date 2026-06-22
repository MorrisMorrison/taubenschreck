async function getJSON(url) { return (await fetch(url)).json(); }
async function post(url) { return (await fetch(url, { method: "POST" })).json(); }

function renderArmed(armed) {
  const b = document.getElementById("armed-badge");
  b.textContent = armed ? "ARMED" : "DISARMED";
  b.className = "badge " + (armed ? "armed" : "disarmed");
}

async function refreshState() { renderArmed((await getJSON("/api/state")).armed); }

async function refreshStats() {
  const s = await getJSON("/api/stats");
  document.getElementById("stat-today").textContent = s.today;
  document.getElementById("stat-total").textContent = s.total;
  document.getElementById("stat-last").textContent = s.last_ts || "–";
}

async function refreshEvents() {
  const { events } = await getJSON("/api/events?limit=25");
  const ul = document.getElementById("events");
  ul.innerHTML = "";
  for (const e of events) {
    const li = document.createElement("li");
    const img = e.snapshot_path
      ? `<img src="/snapshots/${e.snapshot_path.split("/").pop()}" alt="snapshot" />`
      : `<img alt="no snapshot" />`;
    li.innerHTML = `${img}<span>${e.ts} — <b>${e.reason}</b></span>`;
    ul.appendChild(li);
  }
}

async function refreshAll() { await Promise.all([refreshState(), refreshStats(), refreshEvents()]); }

document.getElementById("arm-btn").onclick = async () => { await post("/api/arm"); refreshState(); };
document.getElementById("disarm-btn").onclick = async () => { await post("/api/disarm"); refreshState(); };
document.getElementById("test-fire-btn").onclick = async () => { await post("/api/test-fire"); refreshAll(); };

refreshAll();
setInterval(refreshAll, 3000);
