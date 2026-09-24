"use strict";
/* The GPM console.
 *
 * One more client of the control API: every button here is a call the `pool` CLI can also
 * make. No framework and no build step — the whole page is this file, index.html and
 * console.css, served by the supervisor.
 *
 * The admin key lives in this closure for the tab's lifetime. It is never written to storage
 * and never put in a URL; it goes out as an Authorization header on each call.
 */

let ADMIN_KEY = null;
const state = { status: null, events: [], lastEventId: 0, config: null, screen: "overview" };

// --- the API ---

async function call(method, path, body) {
  const response = await fetch(path, {
    method,
    headers: {
      Authorization: `Bearer ${ADMIN_KEY}`,
      ...(body === undefined ? {} : { "Content-Type": "application/json" }),
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  let payload = null;
  try { payload = await response.json(); } catch { /* some answers have no body */ }
  if (!response.ok) {
    const detail = payload?.detail || payload?.error || `${response.status}`;
    const failure = new Error(detail);
    if (payload?.changes) failure.changes = payload.changes;  // a refused plan explains itself
    throw failure;
  }
  return payload;
}

const api = {
  status: () => call("GET", "/pool/status"),
  events: (limit = 100) => call("GET", `/pool/events?limit=${limit}`),
  leases: () => call("GET", "/pool/leases"),
  plan: () => call("GET", "/pool/plan"),
  account: () => call("GET", "/pool/account"),
  market: (hours = 4, policy) =>
    policy
      ? call("POST", `/pool/market/preview?hours=${hours}&kinds=both`, policy)
      : call("GET", `/pool/market/preview?hours=${hours}&kinds=both`),
  openLease: (body) => call("POST", "/pool/leases", body),
  closeLease: (id) => call("DELETE", `/pool/leases/${id}`),
  tightenLease: (id, body) => call("PATCH", `/pool/leases/${id}`, body),
  prepare: (body) => call("POST", "/pool/hosts/prepare", body),
  hostAction: (id, action) => call("POST", `/pool/hosts/${id}/${action}`),
  down: () => call("POST", "/pool/down"),
  getConfig: () => call("GET", "/pool/config"),
  validateConfig: (text) => call("POST", "/pool/config/validate", { text }),
  planConfig: (text) => call("POST", "/pool/config/plan", { text }),
  applyConfig: (text, version) => call("PUT", "/pool/config", { text, version }),
  configHistory: () => call("GET", "/pool/config/history"),
  rollbackConfig: (version) => call("POST", "/pool/config/rollback", { version }),
  testHost: (host) => call("POST", "/pool/hosts/test", host),
  hostDetail: (id) => call("GET", `/pool/hosts/${encodeURIComponent(id)}`),
  setSearch: (body) => call("PATCH", "/pool/config/rented", body),
  setEngine: (body) => call("PATCH", "/pool/config/engine", body),
  restartEngine: (hostId, applySettings) =>
    call("POST", `/pool/hosts/${hostId}/engine/restart`, { confirm: hostId, apply_settings: applySettings }),
  deleteModel: (hostId, tag) => call("POST", `/pool/hosts/${hostId}/models/delete`, { tag, confirm: tag }),
};

// --- small helpers ---

const el = (tag, attrs = {}, ...children) => {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "html") node.innerHTML = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return node;
};
const money = (n) => (n === null || n === undefined ? "—" : `$${Number(n).toFixed(4)}`);
const rate = (n) => (n === null || n === undefined ? "—" : `$${Number(n).toFixed(3)}/h`);
const clock = (ts) => (ts ? new Date(ts * 1000).toLocaleTimeString() : "—");
const pill = (text, kind) => el("span", { class: `pill ${kind || text || ""}` }, text ?? "—");

// Following one host while it is prepared: its own timeline, what its engine holds right now,
// and how far each download has got. Assembling this by hand from three places is what an
// operator otherwise does while a host sits at "preparing" with nothing to look at.
const hostLink = (id) =>
  el("a", { href: "#", class: "mono", onclick: (e) => { e.preventDefault(); openHost(id); } }, id);

const hostWatch = { id: null, timer: null };

function openHost(id) {
  const dialog = document.getElementById("host-dialog");
  hostWatch.id = id;
  document.getElementById("host-dialog-title").textContent = id;
  document.getElementById("host-dialog-body").replaceChildren(el("p", { class: "muted" }, "Loading…"));
  if (!dialog.open) dialog.showModal();
  document.getElementById("host-dialog-close").onclick = () => closeHost();
  dialog.addEventListener("close", closeHost, { once: true });
  drawHost();
  clearInterval(hostWatch.timer);
  // While a host is being prepared its state changes every few seconds; this is the one view
  // worth following closely, and it is the operator's own machine or their rented one.
  hostWatch.timer = setInterval(drawHost, 3000);
}

function closeHost() {
  clearInterval(hostWatch.timer);
  hostWatch.timer = null;
  hostWatch.id = null;
  const dialog = document.getElementById("host-dialog");
  if (dialog.open) dialog.close();
}

async function drawHost() {
  const id = hostWatch.id;
  if (!id) return;
  let detail;
  try {
    detail = await api.hostDetail(id);
  } catch (error) {
    document.getElementById("host-dialog-body").replaceChildren(
      el("p", { class: "error" }, error.message),
      el("p", { class: "muted" }, "A pool running an older supervisor has no per-host view; restart it to get one."));
    clearInterval(hostWatch.timer);
    return;
  }
  if (hostWatch.id !== id) return;  // closed or switched while we were asking
  // Filtered: replaceChildren turns a null argument into the text "null".
  document.getElementById("host-dialog-body").replaceChildren(...hostPanel(detail).filter(Boolean));
}

function hostPanel(d) {
  const engine = d.engine || {};
  const progress = Object.entries(d.progress || {});
  const facts = [
    ["stage", el("strong", {}, d.stage_detail || d.state)],
    ["state", pill(d.state, d.state === "ready" ? "ok" : "warn")],
    ["kind", `${d.kind}${d.hardware ? " · " + d.hardware : ""}`],
    ["workers", String(d.workers ?? "—")],
  ];
  if (d.bid_hourly !== undefined) {
    facts.push(["rented as", d.interruptible === false
      ? "on demand — a fixed price, cannot be outbid" : "a bid — can be outbid at any moment"]);
    facts.push(["cost", `${rate(d.bid_hourly)} · held ${((d.hours_held || 0) * 60).toFixed(0)} min · spent ${money(d.estimated_spend)} (provider says ${money(d.reported_spend)})`]);
    facts.push(["lease", d.lease_id || "—"]);
  }
  if (d.provider) facts.push(["the provider says", `${d.provider.state}${d.provider.detail ? " · " + d.provider.detail : ""}`]);
  if (d.tunnel) facts.push(["tunnel", `${d.tunnel.up ? "up" : "down"} on :${d.tunnel.local_port} · ${d.tunnel.restarts} restart(s)`]);
  facts.push(["engine", engine.answers
    ? `answers · ${(engine.on_disk || []).length} on disk, ${(engine.loaded || []).length} loaded`
    : el("span", { class: "error" }, `not answering: ${engine.detail || "?"}`)]);

  return [
    el("div", { class: "kv" }, ...facts.flatMap(([k, v]) => [el("div", { class: "k" }, k), el("div", {}, v)])),
    progress.length ? el("div", {},
      el("h2", {}, "Downloads"),
      ...progress.map(([tag, p]) => {
        const share = p.total ? Math.min(1, p.completed / p.total) : 0;
        return el("div", {},
          el("div", { class: "mono" }, `${tag} — ${gigabytes(p.completed)} of ${gigabytes(p.total)}`,
            // Measured here, not the offer's claim — a machine advertising gigabits and
            // delivering a crawl is exactly what gets a host given up.
            p.mbps != null && p.completed < p.total ? el("span", { class: "muted" }, ` · ${p.mbps.toFixed(0)} Mbps`) : null,
            p.attempt > 1 ? el("span", { class: "muted" }, ` · attempt ${p.attempt}`) : null),
          el("div", { class: "burn" }, el("div", { style: `width:${(share * 100).toFixed(1)}%` })));
      })) : null,
    engine.answers ? el("div", {},
      el("h2", {}, "The model set on this host"),
      el("table", {}, el("tbody", {}, (d.required_tags || []).map((tag) => el("tr", {},
        el("td", { class: "mono" }, tag),
        el("td", {}, (engine.loaded || []).includes(tag) ? pill("loaded", "ok")
          : (engine.on_disk || []).includes(tag) ? pill("on disk", "warn")
          : pill("not here yet", "bad")))))),
    ) : null,
    el("h2", {}, "What has happened to this host"),
    (d.events || []).length
      ? el("div", { class: "panel feed" }, (d.events || []).map((e) => el("div", { class: "ev" },
          el("span", { class: "k" }, `${clock(e.ts)}  ${e.kind}  `), e.summary)))
      : el("p", { class: "muted" }, "Nothing recorded for this host yet."),
  ];
}

function confirmAction({ title, body, retype }) {
  const dialog = document.getElementById("confirm-dialog");
  document.getElementById("confirm-title").textContent = title;
  const holder = document.getElementById("confirm-body");
  holder.replaceChildren(typeof body === "string" ? el("p", {}, body) : body);
  const retypeBox = document.getElementById("confirm-retype");
  const input = document.getElementById("confirm-input");
  const ok = document.getElementById("confirm-ok");
  retypeBox.hidden = !retype;
  input.value = "";
  // Loosening is harder than tightening: the new value must be typed again (spec §1).
  const check = () => { ok.disabled = retype ? input.value.trim() !== String(retype) : false; };
  check();
  input.oninput = check;
  dialog.showModal();
  return new Promise((resolve) => {
    dialog.addEventListener("close", () => resolve(dialog.returnValue === "ok"), { once: true });
  });
}

async function run(button, work) {
  const label = button?.textContent;
  if (button) { button.disabled = true; button.textContent = "working…"; }
  try {
    await work();
    await refresh();
  } catch (error) {
    alert(error.message);
  } finally {
    if (button) { button.disabled = false; button.textContent = label; }
  }
}

// --- screens ---

const screens = {};

screens.overview = (status) => {
  const tiers = new Map();
  for (const host of status.hosts) {
    const key = host.priority ?? 0;
    if (!tiers.has(key)) tiers.set(key, []);
    tiers.get(key).push(host);
  }
  for (const host of status.rented) {
    if (!tiers.has(20)) tiers.set(20, []);
    tiers.get(20).push({
      ...host,
      kind: host.interruptible === false ? "rented-on-demand" : "rented-interruptible",
      rented: true,
    });
  }
  // Rented hosts serve like any other once ready, so they count toward capacity. A host that
  // is not ready yet serves nothing, so neither do its workers.
  const serving = [...status.hosts, ...status.rented].filter((h) => h.state === "ready");
  const ready = serving.length;
  const workers = serving.reduce((n, h) => n + (h.workers || 0), 0);
  const busy = serving.reduce((n, h) => n + (h.busy || 0), 0);
  const burn = status.rented.reduce((n, h) => n + (h.bid_hourly || 0), 0);

  return [
    el("h1", {}, "Overview"),
    el("div", { class: "grid" },
      el("div", { class: "panel" }, el("h2", {}, "Capacity"),
        el("div", { class: "stat" }, `${busy} / ${workers}`),
        el("div", { class: "muted" }, `workers busy · ${ready} host(s) ready`)),
      el("div", { class: "panel" }, el("h2", {}, "Rented"),
        el("div", { class: "stat" }, status.rented.length),
        el("div", { class: "muted" }, status.limits.max_hourly_burn == null
          ? `burning ${rate(burn)} · at most ${rate(status.limits.worst_case_hourly)} (${status.limits.max_rented_hosts} × ${rate(status.limits.per_host_ceiling)})`
          : `burning ${rate(burn)} · cap ${rate(status.limits.max_hourly_burn)}`)),
      el("div", { class: "panel" }, el("h2", {}, "Open leases"),
        el("div", { class: "stat" }, status.open_leases.length),
        el("div", { class: "muted" }, status.open_leases.length ? "spending is authorised" : "nothing may be rented")),
    ),
    ...[...tiers.entries()].sort((a, b) => a[0] - b[0]).flatMap(([priority, hosts]) => [
      el("div", { class: "tier" }, `tier ${priority} — ${hosts[0].kind}`),
      el("table", {},
        el("thead", {}, el("tr", {},
          el("th", {}, "Host"), el("th", {}, "State"), el("th", { class: "num" }, "Busy / total"),
          el("th", {}, "Resident"), el("th", { class: "num" }, "Cost"), el("th", {}, ""))),
        el("tbody", {}, hosts.map((host) =>
          el("tr", {},
            el("td", { class: "mono" }, hostLink(host.host_id)),
            el("td", {}, pill(host.state), host.last_error ? el("div", { class: "muted" }, host.last_error) : null),
            el("td", { class: "num" }, `${host.busy ?? 0} / ${host.workers ?? 0}`),
            el("td", { class: "muted mono" }, (host.resident || []).join(", ") || (host.rented ? host.hardware : "—")),
            el("td", { class: "num" }, host.rented ? `${rate(host.bid_hourly)} · ${money(host.estimated_spend)}` : "—"),
            el("td", {}, host.rented
              ? el("div", { class: "row" },
                  el("button", { class: "small", onclick: (e) => run(e.target, () => api.hostAction(host.host_id, "park")) }, "Park"),
                  el("button", { class: "small danger", onclick: (e) => run(e.target, () => api.hostAction(host.host_id, "release")) }, "Release"))
              : null))))),
    ]),
    el("h2", {}, "Leases"), leaseTable(status.open_leases),
    el("h2", {}, "Live feed"), feed(),
  ];
};

const leaseTable = (leases) => leases.length
  ? el("table", {},
      el("thead", {}, el("tr", {},
        el("th", {}, "Lease"), el("th", { class: "num" }, "Workers"), el("th", { class: "num" }, "Hours left"),
        el("th", { class: "num" }, "Spent"), el("th", { class: "num" }, "Cap"), el("th", {}, "Burn-down"), el("th", {}, ""))),
      el("tbody", {}, leases.map((lease) => {
        const fraction = lease.max_spend ? Math.min(1, (lease.spent_enforced_on || 0) / lease.max_spend) : 0;
        return el("tr", {},
          el("td", { class: "mono" }, lease.lease_id, lease.allow_rent ? null : el("div", { class: "muted" }, "may not rent")),
          el("td", { class: "num" }, lease.workers),
          el("td", { class: "num" }, (lease.hours_left ?? 0).toFixed(2)),
          el("td", { class: "num" }, money(lease.spent_enforced_on), lease.estimated_spend !== undefined
            ? el("div", { class: "muted" }, `est ${money(lease.estimated_spend)}`) : null),
          el("td", { class: "num" }, money(lease.max_spend), lease.stops_at !== undefined
            ? el("div", { class: "muted" }, `stops at ${money(lease.stops_at)}`) : null),
          el("td", {}, el("div", { class: `burn ${fraction > 0.8 ? "bad" : fraction > 0.5 ? "warn" : ""}` },
            el("div", { style: `width:${(fraction * 100).toFixed(1)}%` }))),
          el("td", {}, leaseActions(lease)));
      })))
  : el("p", { class: "muted" }, "No open lease, so nothing can be rented.");

// The same three actions wherever a lease is shown — the overview and the Leases screen —
// because an operator who can see a lease should be able to act on it there.
const leaseActions = (lease) => (lease.state && lease.state !== "open") ? null : el("div", { class: "row" },
  el("button", { class: "small", onclick: async (e) => {
    const value = prompt("Tighten the dollar cap to:", String(lease.max_spend));
    if (value === null) return;
    run(e.target, () => api.tightenLease(lease.lease_id, { max_spend: Number(value) }));
  } }, "Tighten"),
  el("button", { class: "small", onclick: (e) => extendLease(e, lease) }, "Extend"),
  el("button", { class: "small", onclick: (e) => run(e.target, () => api.closeLease(lease.lease_id)) }, "Close"));

// Extending a lease in flight: more hours for a host worth keeping, more dollars to pay for
// them, more workers for load the lease is holding back. Raising a limit is loosening wherever
// it appears, so it is typed again (D49) — and the worst case is stated first, because that is
// what the operator is really agreeing to.
//
// **Workers belongs here** (D80). Under dynamic allocation the lease's worker count is the
// ceiling on measured demand, so it is the number that decides whether another host is ever
// rented — and it was the one field this dialog could not change. Live, a pool refusing 70% of
// its requests sat at "the load has cleared" because its lease allowed 5 workers and one
// rented host already supplied 6; the only way to raise it was the CLI.
async function extendLease(event, lease) {
  // One form, not a chain of prompts (D86). Three browser prompts in a row is how the worker
  // ceiling stayed invisible for an hour: an operator answering "hours?" then "dollars?" has
  // no way to see what the lease holds now, or that a third question is coming.
  const field = (label, value, step, why) => {
    const input = el("input", { type: "number", value: String(value), step, style: "width:8rem" });
    return { input, row: el("div", { class: "row" },
      el("label", { style: "min-width:11rem" }, label), input,
      el("span", { class: "muted" }, `now ${value}${why}`)) };
  };
  const hours = field("Run for, in total", lease.max_hours, "0.5", "h");
  const dollars = field("Dollar cap, in total", lease.max_spend, "0.01", "");
  const workers = field("Workers it may reach", lease.workers, "1", "");

  const raisedNote = el("p", { class: "muted" }, "");
  const redraw = () => {
    const raised = [];
    if (Number(hours.input.value) > lease.max_hours) raised.push("hours");
    if (Number(dollars.input.value) > lease.max_spend) raised.push("dollars");
    if (Number(workers.input.value) > lease.workers) raised.push("workers");
    raisedNote.textContent = raised.length
      ? `Raising ${raised.join(", ")} — worst case becomes ${money(Number(dollars.input.value))} over ${hours.input.value}h, up to ${workers.input.value} workers.`
      : "Tightening only. This needs no confirmation.";
  };
  [hours, dollars, workers].forEach((f) => { f.input.oninput = redraw; });
  redraw();

  const raisedValues = () => {
    const out = [];
    if (Number(hours.input.value) > lease.max_hours) out.push(["max_hours", Number(hours.input.value)]);
    if (Number(dollars.input.value) > lease.max_spend) out.push(["max_spend", Number(dollars.input.value)]);
    if (Number(workers.input.value) > lease.workers) out.push(["workers", Number(workers.input.value)]);
    return out;
  };

  // The retype guard has to know the value before the dialog opens, so a raise is confirmed in
  // a second step — the form first, then the retype, which is also where the number is stated.
  const ok = await confirmAction({
    title: `Change ${lease.lease_id}`,
    body: el("div", {}, hours.row, dollars.row, workers.row, raisedNote),
  });
  if (!ok) return;

  const body = {
    max_hours: Number(hours.input.value),
    max_spend: Number(dollars.input.value),
    workers: Number(workers.input.value),
  };
  if (![body.max_hours, body.max_spend, body.workers].every((n) => Number.isFinite(n) && n > 0)) return;

  const raised = raisedValues();
  if (raised.length) {
    const agreed = await confirmAction({
      title: "This loosens a limit",
      body: el("div", {},
        el("p", {}, `Worst case becomes ${money(body.max_spend)} over ${body.max_hours}h, up to ${body.workers} workers — it is spending authority, and hosts held under it keep running.`),
        el("p", { class: "muted" }, raised.map(([k, v]) =>
          k === "max_spend" ? `${k}: ${money(lease[k])} → ${money(v)}`
          : k === "workers" ? `${k}: ${lease[k]} → ${v}`
          : `${k}: ${lease[k]}h → ${v}h`).join(" · "))),
      retype: String(raised[0][1]),
    });
    if (!agreed) return;
    body.confirm = String(raised[0][1]);
  }
  run(event.target, () => api.tightenLease(lease.lease_id, body));
}

const feed = () => el("div", { class: "panel feed" }, state.events.slice(0, 40).map((event) =>
  el("div", { class: "ev" },
    el("span", { class: "k" }, `${clock(event.ts)}  ${event.kind}  `),
    event.summary)));

screens.hosts = (status) => [
  el("h1", {}, "Hosts"),
  el("p", { class: "muted" }, "Configured hosts. Add, edit and disable are configuration changes, so they go through plan on the Configuration screen."),
  el("table", {},
    el("thead", {}, el("tr", {},
      el("th", {}, "Host"), el("th", {}, "Kind"), el("th", {}, "Engine"), el("th", {}, "Transport"), el("th", {}, "State"),
      el("th", { class: "num" }, "Workers"), el("th", {}, "Capabilities"), el("th", {}, "Residency"), el("th", {}, "Tunnel"), el("th", { class: "num" }, "Served"))),
    el("tbody", {}, status.hosts.map((host) => el("tr", {},
      el("td", { class: "mono" }, hostLink(host.host_id)),
      el("td", {}, host.kind),
      // A pool may run a different engine on each machine (D93), and which models this one
      // holds is now a per-host fact too (D89).
      el("td", { class: "mono" }, host.engine || "—",
        (host.holds || []).length ? el("div", { class: "muted" }, `holds ${host.holds.join(", ")}`) : null),
      el("td", {}, host.transport),
      el("td", {}, pill(host.state), host.last_error ? el("div", { class: "muted" }, host.last_error) : null),
      el("td", { class: "num" }, host.workers),
      el("td", { class: "muted" }, (host.capabilities || []).join(", ") || "—"),
      el("td", { class: "muted", title: host.residency === "on_demand"
          ? "routable once the model set is on disk; the engine loads a model on first use and may evict it"
          : "routable only while the whole model set is loaded" },
        host.residency === "on_demand" ? "on demand" : "pinned"),
      el("td", {}, host.tunnel ? pill(host.tunnel.up ? "up" : "down", host.tunnel.up ? "ok" : "bad") : "—",
        host.tunnel ? el("div", { class: "muted mono" }, `:${host.tunnel.local_port} · ${host.tunnel.restarts} restart(s)`) : null),
      el("td", { class: "num" }, host.requests_served ?? 0))))),
  ...status.hosts.filter((host) => host.agent).map(agentPanel),
  el("h2", {}, "Test connection"),
  hostTestForm(),
];

// What the agent on the machine reports — read, never typed. A silent agent is said to be
// silent; the host keeps serving and nothing is inferred from it.
const gigabytes = (bytes) => (bytes === null || bytes === undefined ? "—" : `${(bytes / 1e9).toFixed(1)} GB`);
function agentPanel(host) {
  const agent = host.agent;
  const facts = agent.facts;
  const head = el("h2", {}, `Agent on ${host.host_id} `,
    pill(agent.reachable ? "answers" : "silent", agent.reachable ? "ok" : "warn"));
  if (!facts) {
    return el("div", { class: "panel" }, head,
      el("p", { class: "muted" }, `${agent.detail || "no answer yet"} · key ${agent.key === "set" ? "set ✓" : "missing ✗"} · ${agent.url}`),
      el("p", { class: "muted" }, "The host is still verified by its engine and keeps serving. Nothing is assumed about the machine."));
  }
  const accelerators = facts.accelerators === null ? "could not tell"
    : facts.accelerators.length ? facts.accelerators.map((a) => `${a.name} · ${gigabytes(a.memory_bytes)}${a.unified ? " unified" : ""}`).join("; ")
    : "none found";
  const engine = facts.engine || {};
  return el("div", { class: "panel" }, head,
    agent.capability_conflict ? el("p", { class: "error" }, agent.capability_conflict) : null,
    el("div", { class: "kv" },
      el("div", { class: "k" }, "machine"), el("div", {}, `${facts.os} · ${facts.arch}`),
      el("div", { class: "k" }, "accelerators"), el("div", {}, accelerators),
      el("div", { class: "k" }, "capabilities"), el("div", {},
        (host.capabilities || []).join(", ") || "—",
        el("span", { class: "muted" }, (host.configured_capabilities || []).length ? " — as configured" : " — found by the agent")),
      el("div", { class: "k" }, "memory"), el("div", {}, `${gigabytes(facts.memory.available_bytes)} available of ${gigabytes(facts.memory.total_bytes)}`),
      el("div", { class: "k" }, "disk for models"), el("div", {}, `${gigabytes(facts.disk.free_bytes)} free`, el("span", { class: "muted mono" }, `  ${facts.disk.path}`)),
      el("div", { class: "k" }, "engine"), el("div", {}, engine.answers
        ? `${engine.name} ${engine.version || ""} · ${(engine.models_on_disk || []).length} model(s) on disk, ${(engine.models_loaded || []).length} loaded`
        : el("span", { class: "error" }, `${engine.name}: ${engine.detail || "not answering"}`)),
      el("div", { class: "k" }, "owner allows"), el("div", { class: "muted" },
        `delete models: ${facts.allows.delete ? "yes" : "no"} · restart engine: ${facts.allows.restart ? "yes" : "no command set"}`),
      el("div", { class: "k" }, "agent"), el("div", { class: "muted" }, `v${facts.agent.version} · last answered ${clock(agent.asked_at)}`)),
    engineControl(host, agent, facts),
    agentModels(host, agent, facts));
}

// The engine has to run as many requests at once as the host has workers, and hold the whole
// model set together, or the pool's numbers are fiction. With an agent the pool can set that —
// through a command and a file path the machine's owner wrote, and only when you press this.
function engineControl(host, agent, facts) {
  const wanted = agent.wanted_engine_settings;
  const inForce = facts.engine_settings;
  const differs = !inForce || inForce.workers !== wanted.workers || (inForce.models_held ?? 0) < wanted.models_held;
  const describe = (s) => (s ? `${s.workers ?? "?"} at once · ${s.models_held ?? "?"} model(s) held` : "never set through the agent");
  const restart = (applySettings) => async (e) => {
    const ok = await confirmAction({
      title: applySettings ? `Apply the pool's settings and restart the engine on ${host.host_id}?` : `Restart the engine on ${host.host_id}?`,
      body: `The engine stops and starts again, using the command ${host.host_id}'s owner wrote on that machine. Requests in flight there fail over to another host, or fail.`
        + (applySettings ? ` It will start with ${describe(wanted)}.` : ""),
      retype: host.host_id,
    });
    if (!ok) return;
    const button = e.target; const label = button.textContent; button.disabled = true; button.textContent = "restarting…";
    try {
      const result = await api.restartEngine(host.host_id, applySettings);
      alert(result.engine_answers
        ? `Restarted (exit ${result.restart_exit_code}). The engine is answering.`
        : `The command ran (exit ${result.restart_exit_code}) but the engine is NOT answering.\n\n${result.restart_output || ""}`);
      await refresh();
    } catch (error) { alert(error.message); } finally { button.disabled = false; button.textContent = label; }
  };
  return el("div", {},
    el("h2", {}, "Engine settings"),
    el("div", { class: "kv" },
      el("div", { class: "k" }, "the pool needs"), el("div", {}, describe(wanted)),
      el("div", { class: "k" }, "set on the machine"), el("div", {}, describe(inForce), " ",
        differs ? pill("differs", "warn") : pill("matches", "ok"))),
    facts.allows.restart
      ? el("div", { class: "row" },
          facts.allows.engine_settings
            ? el("button", { class: differs ? "primary" : "", onclick: restart(true) }, "Apply the pool's settings and restart")
            : null,
          el("button", { onclick: restart(false) }, "Restart engine"))
      : null,
    facts.allows.restart && facts.allows.engine_settings ? null : el("p", { class: "muted" },
      !facts.allows.restart
        ? "To let the pool restart this engine, the machine's owner adds a restart_command (an argument list) to the agent's settings there. The pool can ask for it to be run; it can never say what it is."
        : "To let the pool set these, the machine's owner adds an engine_env_file path to the agent's settings, and points the engine's service at that file."));
}

// What the machine holds against what the pool asked it to hold. The agent pulls what is
// missing; it never deletes. Deleting is the button below, and only that.
function agentModels(host, agent, facts) {
  if (!agent.manages_models) return el("p", { class: "muted" }, "This agent reports only: manage_models is off for this host.");
  const models = agent.models;
  if (!models) return el("p", { class: "muted" }, "Waiting for the agent's first answer about models.");
  const stateOf = (model) => {
    if (model.pulling) {
      const { completed_bytes: done, total_bytes: total } = model.pulling;
      const share = total ? Math.min(1, done / total) : 0;
      return el("div", {}, pill("pulling", "warn"), ` ${gigabytes(done)} of ${gigabytes(total)}`,
        el("div", { class: "burn" }, el("div", { style: `width:${(share * 100).toFixed(1)}%` })));
    }
    if (model.error) return el("div", {}, pill("failed", "bad"), el("div", { class: "muted" }, model.error));
    if (model.loaded) return pill(model.pinned_by_agent ? "loaded · pinned by the agent" : "loaded", "ok");
    if (model.on_disk) return pill(models.residency === "on_demand" ? "on disk · loads on use" : "on disk · loading", "ok");
    return pill("not on disk yet", "warn");
  };
  const mayDelete = facts.allows.delete;
  return el("div", {},
    el("h2", {}, "Models the pool asked this machine to hold ", el("span", { class: "muted" }, `· ${models.residency === "on_demand" ? "on demand" : "pinned"}`)),
    el("table", {}, el("tbody", {}, models.models.map((model) => el("tr", {},
      el("td", { class: "mono" }, model.tag), el("td", { class: "num muted" }, gigabytes(model.size_bytes)), el("td", {}, stateOf(model)))))),
    el("p", { class: "muted" }, `${gigabytes(models.free_disk_bytes)} free; pulls stop before it falls under this machine's ${gigabytes(models.min_free_disk_bytes)} floor, which its owner sets.`),
    el("h2", {}, "Other models on this machine ", el("span", { class: "muted" }, "· not named by this pool; never touched unless you delete one")),
    models.surplus.length ? el("table", {}, el("tbody", {}, models.surplus.map((model) => el("tr", {},
      el("td", { class: "mono" }, model.tag),
      el("td", { class: "num muted" }, gigabytes(model.size_bytes)),
      el("td", { class: "muted" }, model.loaded ? "loaded now" : ""),
      el("td", {}, mayDelete
        ? el("button", { class: "small danger", onclick: async (e) => {
            const ok = await confirmAction({
              title: `Delete ${model.tag} from ${host.host_id}?`,
              body: `This removes ${gigabytes(model.size_bytes)} from that machine's disk. It cannot be undone; the model would have to be downloaded again. Other software on the machine may be using it.`,
              retype: model.tag,
            });
            if (ok) run(e.target, () => api.deleteModel(host.host_id, model.tag));
          } }, "Delete")
        : null)))))
      : el("p", { class: "muted" }, "None."),
    mayDelete ? null : el("p", { class: "muted" }, "This machine's owner has switched deletion off in the agent's own settings; the pool cannot change that."));
}

// A serialising engine is the one failure the pool can put right itself — on a host that runs
// an agent whose owner has allowed it. Say which hosts those are, rather than leave the
// operator with a variable name to go and set by hand.
function engineSettingsHint() {
  const withAgent = (state.status?.hosts || []).filter((host) => host.agent?.facts);
  const able = withAgent.filter((host) => host.agent.facts.allows.engine_settings);
  if (able.length) {
    return el("p", {}, "The pool can fix this itself on ", el("strong", {}, able.map((h) => h.host_id).join(", ")),
      ": under that host's agent above, press ", el("strong", {}, "Apply the pool's settings and restart"), ".");
  }
  return el("p", { class: "muted" }, withAgent.length
    ? "A host's agent could fix this for you, once that machine's owner adds a restart_command and an engine_env_file to the agent's settings."
    : "With gpm-agent running on the host, the pool could set this and restart the engine for you.");
}

function hostTestForm() {
  const output = el("div", { class: "muted" }, "Checks an unsaved host definition: reach, authenticate, the whole model set resident together, capabilities, and whether the engine actually serves requests in parallel. Saves nothing.");
  const url = el("input", { placeholder: "http://127.0.0.1:11434", size: 34 });
  const workers = el("input", { type: "number", value: "2", min: "1", max: "16", style: "width:5rem" });
  return el("div", { class: "panel" },
    el("div", { class: "row" },
      el("label", {}, "Engine URL ", url),
      el("label", {}, "Workers wanted ", workers),
      el("button", { class: "primary", onclick: async (e) => {
        const button = e.target; const label = button.textContent;
        button.disabled = true; button.textContent = "testing…";
        try {
          const result = await api.testHost({ base_url: url.value.trim(), workers: Number(workers.value) });
          output.replaceChildren(el("table", {},
            el("tbody", {}, result.steps.map((step) => el("tr", {},
              el("td", {}, pill(step.ok ? "pass" : "fail", step.ok ? "ok" : "bad")),
              el("td", {}, step.name),
              el("td", { class: "muted" }, step.detail || ""))))),
            el("p", {}, `Workers that would apply: ${result.workers}`),
            result.steps.some((step) => step.fix === "engine_settings") ? engineSettingsHint() : null);
        } catch (error) {
          output.replaceChildren(el("p", { class: "error" }, error.message));
        } finally { button.disabled = false; button.textContent = label; }
      } }, "Test connection")),
    output);
}

screens.rented = async (status) => {
  if (!status.provider) return [el("h1", {}, "Rented capacity"), el("p", { class: "muted" }, "This pool has no rented capacity configured, so it cannot spend.")];
  const [account, market] = await Promise.all([
    api.account().catch((e) => ({ error: e.message })),
    api.market(1).catch((e) => ({ error: e.message })),
  ]);
  const capabilities = Object.entries(status.provider.capabilities).filter(([, on]) => on).map(([name]) => name);
  return [
    el("h1", {}, "Rented capacity"),
    el("div", { class: "grid" },
      el("div", { class: "panel" }, el("h2", {}, "Provider"),
        el("div", { class: "kv" },
          el("div", { class: "k" }, "name"), el("div", {}, status.provider.name),
          el("div", { class: "k" }, "credential"), el("div", {}, account.error ? pill("error", "bad") : pill(account.credential_valid ? "valid ✓" : "invalid ✗", account.credential_valid ? "ok" : "bad")),
          el("div", { class: "k" }, "credit left"), el("div", {}, account.error ? account.error : money(account.credit_remaining)),
          el("div", { class: "k" }, "can"), el("div", { class: "muted" }, capabilities.join(", ")),
          el("div", { class: "k" }, "cap margin"), el("div", {}, `${(status.provider.cap_safety_margin * 100).toFixed(0)}%`,
            el("span", { class: "muted" }, status.provider.capabilities.reports_charges ? " — narrows once a charge is reported" : " — wider: this provider reports no charges")))),
      el("div", { class: "panel" }, el("h2", {}, "Limits"),
        el("div", { class: "kv" },
          el("div", { class: "k" }, "max rented hosts"), el("div", {}, hostLimitControl(status.limits.max_rented_hosts)),
          el("div", { class: "k" }, "per-host ceiling"), el("div", {}, rate(status.limits.per_host_ceiling)),
          el("div", { class: "k" }, "overall cap"), el("div", {}, status.limits.max_hourly_burn == null
            ? el("span", { class: "muted" }, `none — bounded at ${rate(status.limits.worst_case_hourly)} by hosts × ceiling`)
            : rate(status.limits.max_hourly_burn))),
        el("p", { class: "muted" }, "Raising either is a configuration change that must be retyped to confirm.")),
      enginePanel(status.engine),
      preparePanel(),
    ),
    ...searchSection(market),
    ...allocationSection(market),
    ...engineSection(status),
    ...teardownSection(market),
    ...marketSection(market),
    el("h2", {}, "Rented and parked hosts"),
    status.rented.length ? el("table", {},
      el("thead", {}, el("tr", {},
        el("th", {}, "Host"), el("th", {}, "State"), el("th", {}, "Serving"), el("th", {}, "Machine"), el("th", {}, "Rented as"), el("th", { class: "num" }, "Price"),
        el("th", { class: "num" }, "Storage"), el("th", { class: "num" }, "Held"), el("th", { class: "num" }, "Spend"), el("th", {}, ""))),
      el("tbody", {}, status.rented.map((host) => el("tr", {},
        el("td", { class: "mono" }, hostLink(host.host_id)),
        el("td", {}, pill(host.state)),
        // What this machine was bought to serve, and what runs it (D93, D94). A pool buying
        // the wrong thing used to look exactly like one buying the right thing.
        el("td", {}, boughtFor(host)),
        el("td", { class: "muted" }, `${host.machine} · ${host.hardware || ""}`),
        el("td", {}, host.interruptible === false ? pill("on demand", "ok") : pill("bid", "warn")),
        el("td", { class: "num" }, rate(host.bid_hourly)),
        el("td", { class: "num" }, rate(host.storage_hourly)),
        el("td", { class: "num" }, `${(host.hours_held ?? 0).toFixed(2)}h`),
        el("td", { class: "num" }, money(host.estimated_spend), el("div", { class: "muted" }, `rep ${money(host.reported_spend)}`)),
        el("td", {}, el("div", { class: "row" },
          el("button", { class: "small", onclick: (e) => run(e.target, () => api.hostAction(host.host_id, "park")) }, "Park"),
          el("button", { class: "small danger", onclick: (e) => run(e.target, () => api.hostAction(host.host_id, "release")) }, "Release")))))))
      : el("p", { class: "muted" }, "Nothing rented."),
  ];
};

// The market, refreshed in place: by the button, and on its own every minute while it is on
// screen. Only this panel is replaced — re-rendering the whole screen would wipe whatever is
// being typed into the Prepare form beside it. And it only asks while it is visible: each
// refresh is a real call to the provider, which is what got the pool rate-limited (D44). A
// refresh during that back-off costs the provider nothing — it reports why, and waits.
// Every parameter the offer search uses, editable here. Trying values is free — the preview
// runs the real pipeline against the live market and saves nothing — and saving goes through
// the file, the plan, and the retype rule, exactly as the Configuration screen does (D51).
const SEARCH_FIELDS = [
  ["offer_policy", "min_gpu_memory_gb", "range", "the card must have at least this much memory", 0, 200, 4],
  ["offer_policy", "min_disk_gb", "range", "the machine must offer at least this much disk", 0, 500, 10],
  ["offer_policy", "max_all_in_hourly", "range", "the most this pool will pay per host, per hour", 0, 20, 0.05],
  ["offer_policy", "max_all_in_per_gpu", "range", "and per accelerator — a multi-GPU machine is judged by the card, not the bill", 0, 10, 0.05],
  ["offer_policy", "max_download_per_gb", "range", "the most it will pay per GB downloaded", 0, 0.5, 0.005],
  ["offer_policy", "min_download_mbps", "range", "slower than this and the model set takes too long", 0, 10000, 100],
  ["offer_policy", "min_reliability", "range", "the provider's own score, 0 to 1", 0, 1, 0.01],
  ["offer_policy", "min_driver_version", "text", "the accelerator driver this engine image needs — below it the card sits idle and the CPU serves"],
  ["offer_policy", "verified_only", "checkbox", "only machines the provider has verified"],
  ["offer_policy", "exclude_hardware", "list", "refused by name, case-insensitive"],
  ["offer_policy", "avoid_machines", "list", "machine ids to skip — one that keeps failing, say"],
  ["bidding", "bid_ceiling", "range", "never bid above this, whatever a strategy returns", 0, 20, 0.05],
  ["bidding", "premium", "range", "added to the market floor when bidding", 0, 2, 0.01],
  ["bidding", "on_demand_crossover", "range", "past this fraction of the on-demand price, do not bid", 0, 1, 0.01],
  ["bidding", "attempts", "range", "offers to try in one pass before giving up", 1, 20, 1],
];

// How capacity is allocated, editable on the same screen for the same reason (D74): two
// opt-in features that spend money should be visible and changeable where renting is watched,
// not only in a file on the supervisor's machine.
const ALLOCATION_FIELDS = [
  ["dynamic", "target_utilisation", "range", "rent before saturation, not at it: 0.75 keeps a quarter spare", 0.1, 1, 0.05],
  ["dynamic", "window_s", "range", "load must hold this long before the first host is bought", 0, 600, 10],
  ["dynamic", "ramp_factor", "range", "each round asks for this many times the last: 1, 2, 4 …", 1, 4, 0.5],
  ["dynamic", "ramp_backoff_s", "range", "and waits this long after the last round has landed", 0, 900, 30],
  ["dynamic", "max_round", "range", "the most one round may add, however long the load lasts", 1, 16, 1],
  ["dynamic", "min_hosts", "range", "a warm floor kept while a lease is open; 0 spends nothing when quiet", 0, 8, 1],
  ["workers_auto", "enabled", "checkbox", "each host finds its own worker count while it serves"],
  ["workers_auto", "max", "range", "the most any host's engine is launched to run at once", 1, 64, 1],
  ["workers_auto", "min_gain", "range", "a step up must raise throughput by this much to count", 0, 1, 0.01],
  ["workers_auto", "slow_host_factor", "range", "service time this far over the pool's median steps a host down", 1, 5, 0.1],
];

// When a host is given up, and how hard the pool tries first (D86). Sliders, because every
// one of these is a quantity with a sane range and an operator should see where in that range
// they are — not type a number into an empty box and hope.
//   [section, key, kind, why, min, max, step]
const TEARDOWN_FIELDS = [
  ["teardown", "idle_minutes", "range", "no traffic for this long and a host is paused", 0.5, 30, 0.5],
  ["teardown", "destroy_idle_minutes", "range", "and destroyed at this; blank means 2.5x the pause", 0, 120, 1],
  ["teardown", "max_starting_minutes", "range", "its engine never answered — give it up rather than bill for the whole window", 1, 30, 1],
  ["teardown", "max_preparing_minutes", "range", "started, but never finished its model set", 5, 90, 5],
  ["teardown", "drain_timeout_s", "range", "how long in-flight requests get once a host is going", 30, 900, 30],
  ["teardown", "deadman_minutes", "range", "the host ends itself after this with no heartbeat and no traffic", 5, 120, 5],
  ["teardown", "deadman_action", "choice", "what it does when it fires", ["destroy", "stop"]],
  ["teardown", "park_when_idle", "checkbox", "pause an unused host rather than destroy it, keeping its models"],
  ["teardown", "max_park_hours", "range", "a parked host is never kept longer than this", 1, 168, 1],
  ["teardown", "min_pull_mbps", "range", "a download slower than this gives the host up", 0, 500, 10],
  ["teardown", "slow_pull_grace_s", "range", "after it has been that slow for this long", 30, 600, 30],
  ["teardown", "pull_attempts", "range", "tries per model before the host is given up", 1, 10, 1],
  ["teardown", "pull_retry_after_s", "range", "wait before the second try; it doubles after", 1, 120, 1],
  ["teardown", "avoid_failed_machine_minutes", "range", "a machine that just failed is skipped for this long", 0, 240, 10],
  ["teardown", "max_hours_without_deadman", "range", "the longest lease allowed where nothing on the host can stop it billing", 0.5, 12, 0.5],
];

const search = { inputs: {}, saved: null, message: "", mode: null };
const teardown = { inputs: {}, saved: null, message: "" };

// A slider an operator can also type into: the handle shows where in the range a value sits,
// the box says exactly what it is. Neither alone is enough — a slider cannot express 12.5 and
// a bare number box does not say whether 30 is high or low.
function slider(value, min, max, step, onchange) {
  const shown = value === null || value === undefined ? "" : String(value);
  const range = el("input", { type: "range", min: String(min), max: String(max), step: String(step),
                              value: shown === "" ? String(min) : shown, style: "width:11rem" });
  const box = el("input", { type: "number", step: String(step), value: shown, style: "width:6rem" });
  range.oninput = () => { box.value = range.value; onchange && onchange(); };
  box.oninput = () => { if (box.value !== "") range.value = box.value; onchange && onchange(); };
  return { row: el("span", { class: "row" }, range, box), read: () => (box.value === "" ? null : Number(box.value)) };
}

// Every configuration row on this screen is built here, so the sections cannot drift apart in
// look or in how their values are read back. `range` is the default for a number with known
// bounds: the handle shows where in the range a value sits, the box says exactly what it is.
function controlFor(kind, value, a, b, c) {
  if (kind === "checkbox") {
    const input = el("input", { type: "checkbox", ...(value ? { checked: true } : {}) });
    return { control: input, read: () => input.checked };
  }
  if (kind === "choice") {
    const input = el("select", {}, ...a.map((v) =>
      el("option", { value: v, ...(String(value) === v ? { selected: true } : {}) }, v)));
    return { control: input, read: () => input.value };
  }
  if (kind === "list" || kind === "text") {
    const shown = value === null || value === undefined ? "" : (kind === "list" ? value.join(", ") : String(value));
    const input = el("input", { type: "text", style: "width:17rem", value: shown });
    return {
      control: input,
      read: () => {
        const raw = input.value.trim();
        if (kind === "list") return raw ? raw.split(",").map((t) => t.trim()).filter(Boolean) : [];
        return raw === "" ? null : raw;
      },
    };
  }
  if (kind === "number" || a === undefined) {
    // No bounds to put a handle on — a plain box, rather than a slider that would invent them.
    const shown = value === null || value === undefined ? "" : String(value);
    const input = el("input", { type: "number", step: "any", style: "width:9rem", value: shown });
    return { control: input, read: () => (input.value.trim() === "" ? null : Number(input.value)) };
  }
  const made = slider(value, a, b, c);
  return { control: made.row, read: made.read };
}

function configRow(key, label, made, why) {
  return el("tr", {},
    el("td", { class: "mono" }, label),
    el("td", {}, made.control),
    el("td", { class: "muted" }, why));
}

// What would actually be started on a machine this pool rents (D92). Until this existed
// neither the engine nor the image was visible anywhere in the console, and a pool set to one
// engine with an image built for another looked exactly like a correct one — until a host had
// been bought, started the wrong server, and never answered.
// What a rented host was bought to serve, with the engine that serves it. A host bought before
// the pool assigned models holds the whole rented set, and says so rather than showing nothing.
function boughtFor(host) {
  const models = host.bought_for || [];
  const engine = host.engine ? el("div", { class: "muted mono" }, host.engine) : null;
  if (!models.length) {
    return el("div", {}, el("span", { class: "muted" }, "the whole rented set"), engine);
  }
  return el("div", {}, ...models.map((name) => el("div", { class: "mono" }, name)), engine);
}

function enginePanel(engine) {
  if (!engine) return el("div", { class: "panel" }, el("h2", {}, "Engine"),
    el("p", { class: "muted" }, "This pool's supervisor does not report its engine yet; restart it to see it here."));
  const images = engine.images || [];
  const rows = images.map((i) => el("tr", {},
    el("td", { class: "mono" }, i.image),
    el("td", {}, i.min_driver
      ? el("span", {}, "driver ", el("span", { class: "mono" }, i.min_driver), "+")
      : el("span", { class: "muted" }, "any driver the policy allows")),
    el("td", { class: "muted" }, i.note || "")));
  const several = (engine.in_use || []).length > 1;
  return el("div", { class: "panel" }, el("h2", {}, "Engine"),
    el("div", { class: "kv" },
      el("div", { class: "k" }, "pool default"), el("div", { class: "mono" }, engine.name),
      el("div", { class: "k" }, "rented hosts"), el("div", { class: "mono" }, engine.rented || engine.name,
        several ? el("div", { class: "muted" }, `this pool runs ${(engine.in_use || []).join(" and ")}`) : null),
      el("div", { class: "k" }, "port"), el("div", { class: "mono" }, String(engine.port)),
      el("div", { class: "k" }, "model set"), el("div", {}, engine.models_per_host === "all"
        ? (engine.proxy
            ? "every host holds the whole set — one engine process per model, behind a router on the machine"
            : "every host holds the whole set")
        : "each host holds what it declares; the pool buys a host per model")),
    el("table", {}, el("tbody", {}, ...rows)),
    el("p", { class: "muted" }, images.length > 1
      ? "A machine is rented with the first build its driver can run; one that can run none is refused before it is bid on."
      : "One build for every machine. The driver floor in the search is what keeps an unusable machine out."));
}

// Which engine rented hosts run, and how the pool's models are placed on them (D98). One write,
// because none of these can change alone: an engine switched without its builds, images or start
// command rents machines that can never serve — and the file refuses every half-way state.
const engineEdit = { draft: null, message: "" };

// Suggested vLLM builds of the engine, newest first: a machine gets the first its driver runs (D92).
const VLLM_IMAGE_SUGGESTIONS = [
  { image: "vastai/vllm:v0.29.0-cuda-13.0", min_driver: "580", note: "newest; needs a recent driver" },
  { image: "vastai/vllm:v0.29.0-cuda-12.9", min_driver: "550", note: "for older drivers" },
];

const PLACEMENTS = [
  ["all", "every host holds every model — one engine process",
   (engine) => engine === "vllm" ? "vLLM serves one model per process; choose one of the other two" : null],
  ["all_proxy", "every host holds every model — one vLLM per model, behind a router on the machine",
   (engine) => engine !== "vllm" ? "only for vLLM; Ollama already holds several models in one process" : null],
  ["declared", "one model per host — a pool of hosts for each model", () => null],
];

function engineDraft(status) {
  const e = status.engine || {};
  const rented = e.rented || e.name || "ollama";
  const placement = e.models_per_host === "declared" ? "declared" : (e.proxy ? "all_proxy" : "all");
  const builds = {};
  for (const name of status.model_set || []) {
    const hit = ((status.catalog || {})[name] || []).find((v) => v.engine === rented);
    builds[name] = hit ? hit.tag : "";
  }
  const images = (e.images || []).filter((i) => i.min_driver);
  return {
    rented, placement, builds,
    rented_models: e.rented_models || [],
    images: images.length ? images.map((i) => ({ ...i })) : null,
    image: e.image || "",
    engine_start: e.engine_start || "",
  };
}

function engineSection(status) {
  if (!status.engine || !status.provider) return [];
  if (!engineEdit.draft) engineEdit.draft = engineDraft(status);
  const box = el("div", { class: "panel" });
  const draw = () => {
    box.replaceChildren(...engineRows(status, box, draw));
  };
  draw();
  return [el("h2", {}, "Engine and placement on rented hosts"), box];
}

function engineRows(status, box, draw) {
  const d = engineEdit.draft;
  const available = (status.engine.available || ["ollama"]);
  const vllm = d.rented === "vllm";
  const invalid = Object.fromEntries(PLACEMENTS.map(([v, , why]) => [v, why(d.rented)]));
  // When the engine rules out the current placement, land on a model to a host: each engine
  // then has the whole card, which is where a batching engine's throughput comes from.
  if (invalid[d.placement]) d.placement = !invalid.declared ? "declared" : PLACEMENTS.find(([v]) => !invalid[v])[0];

  const engineSelect = el("select", { onchange: (ev) => {
    d.rented = ev.target.value;
    // Offer what the chosen engine needs, starting from what the pool already has for it.
    const fresh = engineDraft({ ...status, engine: { ...status.engine, rented: d.rented } });
    d.builds = fresh.builds;
    if (d.rented === "vllm") {
      d.images = d.images || VLLM_IMAGE_SUGGESTIONS.map((i) => ({ ...i }));
      if (/ollama/i.test(d.engine_start)) d.engine_start = "";
    }
    draw();
  } }, ...available.map((name) => el("option", { value: name, ...(d.rented === name ? { selected: true } : {}) }, name)));

  const placementSelect = el("select", { onchange: (ev) => { d.placement = ev.target.value; draw(); } },
    ...PLACEMENTS.map(([value, label]) => el("option", {
      value, ...(d.placement === value ? { selected: true } : {}), ...(invalid[value] ? { disabled: true } : {}),
    }, invalid[value] ? `${label} (${invalid[value]})` : label)));

  const rows = [
    configRow("engine", "rented engine", { control: engineSelect },
      "what the machines this pool buys run; hosts you configured keep their own"),
    configRow("placement", "placement", { control: placementSelect },
      d.placement === "all_proxy" ? "each machine's memory is split between the models, so the largest gets less cache than it would alone"
        : d.placement === "declared" ? "the pool buys a machine for whichever model is short, and never takes the last one serving a model"
        : "any ready host serves any request"),
  ];

  if (d.placement === "declared") {
    const boxes = (status.model_set || []).map((name) => {
      const input = el("input", { type: "checkbox", ...(d.rented_models.includes(name) ? { checked: true } : {}),
        onchange: (ev) => {
          d.rented_models = ev.target.checked ? [...d.rented_models, name] : d.rented_models.filter((m) => m !== name);
          draw();  // a model now rented for needs its build asked for, one no longer does not
        } });
      return el("label", { class: "row" }, input, el("span", { class: "mono" }, name));
    });
    rows.push(configRow("rented_models", "rent hosts for", { control: el("div", {}, ...boxes) },
      "models your configured hosts hold need not be rented for; a model nobody holds is refused on save"));
  }

  const renting = d.placement === "declared" ? d.rented_models : (status.model_set || []);
  for (const name of status.model_set || []) {
    if (d.placement === "declared" && !renting.includes(name)) continue;
    const input = el("input", { type: "text", style: "width:22rem", value: d.builds[name] || "",
      placeholder: vllm ? "owner/name of the model on the hub" : "the engine's tag for this model",
      oninput: (ev) => { d.builds[name] = ev.target.value; } });
    rows.push(configRow(`build ${name}`, `${d.rented} build of ${name}`, { control: input },
      vllm ? "the repository vLLM fetches — the same model under the name this engine knows it by"
        : "blank keeps what the catalog already has"));
  }

  if (vllm) {
    const list = el("div", {}, ...(d.images || []).map((img, i) => el("div", { class: "row" },
      el("input", { type: "text", style: "width:18rem", value: img.image,
        oninput: (ev) => { d.images[i].image = ev.target.value; } }),
      el("span", { class: "muted" }, "driver ≥"),
      el("input", { type: "text", style: "width:5rem", value: img.min_driver,
        oninput: (ev) => { d.images[i].min_driver = ev.target.value; } }),
      el("button", { class: "small", onclick: () => { d.images.splice(i, 1); draw(); } }, "Remove"))),
      el("button", { class: "small", onclick: () => { d.images.push({ image: "", min_driver: "" }); draw(); } }, "Add a build"));
    rows.push(configRow("images", "images, newest first", { control: list },
      "a machine gets the first its driver can run; one that can run none is never bid on"));
  } else {
    rows.push(configRow("image", "image", { control: el("input", { type: "text", style: "width:18rem", value: d.image,
      oninput: (ev) => { d.image = ev.target.value; } }) }, "pinned, never a floating tag"));
  }

  rows.push(configRow("engine_start", "start command", {
    control: el("input", { type: "text", style: "width:26rem", value: d.engine_start,
      placeholder: vllm ? "blank: the agent's own vllm-start (recommended)" : "how the image's engine is started, if it does not start itself",
      oninput: (ev) => { d.engine_start = ev.target.value; } }),
  }, vllm ? "blank lets the pool start vLLM on what its agent downloaded" : "runs after the dead-man timer is armed"));

  const note = el("span", { class: "muted" }, engineEdit.message);
  return [
    el("table", {}, el("tbody", {}, ...rows)),
    el("div", { class: "row" },
      el("button", { class: "primary", onclick: (ev) => saveEngine(ev, note) }, "Save to configuration"),
      el("button", { class: "small", onclick: () => { engineEdit.draft = engineDraft(status); engineEdit.message = ""; draw(); } }, "Reset"),
      note),
    el("p", { class: "muted" }, "Saved as one change to the file, with its comments kept. Hosts already running keep what they were started with; new ones use this."),
  ];
}

async function saveEngine(event, note) {
  const d = engineEdit.draft;
  const body = {
    rented_engine: d.rented, placement: d.placement,
    rented_models: d.placement === "declared" ? d.rented_models : [],
    builds: Object.fromEntries(Object.entries(d.builds).filter(([, tag]) => (tag || "").trim())),
    engine_start: d.engine_start,
  };
  if (d.rented === "vllm") body.images = (d.images || []).filter((i) => (i.image || "").trim());
  else { body.images = []; if ((d.image || "").trim()) body.image = d.image.trim(); }
  const button = event.target;
  button.disabled = true;
  try {
    let answer;
    try {
      answer = await api.setEngine(body);
    } catch (error) {
      const changes = error.changes || [];
      const retype = changes.find((c) => c.requires_retype);
      if (!retype) throw error;
      const ok = await confirmAction({
        title: "This loosens a limit",
        body: el("div", {}, ...changes.map((c) => el("p", {}, c.detail))),
        retype: retype.value,
      });
      if (!ok) return;
      answer = await api.setEngine({ ...body, confirm: retype.value });
    }
    engineEdit.draft = null;
    engineEdit.message = ` saved · ${(answer.changes || []).length} change(s) applied`;
    note.textContent = engineEdit.message;
    await refresh();
  } catch (error) {
    // The file's own rules say what is wrong, in words written for an operator.
    engineEdit.message = ` not saved: ${error.message}`;
    note.textContent = engineEdit.message;
  } finally {
    button.disabled = false;
  }
}

function teardownSection(market) {
  const saved = (market.saved || {}).teardown;
  teardown.saved = saved;
  teardown.inputs = {};
  if (!saved) {
    return [
      el("h2", {}, "When a host is given up"),
      el("p", { class: "muted" }, "This pool's supervisor does not report its tear-down settings yet; restart it to edit them here."),
    ];
  }
  const rows = TEARDOWN_FIELDS.map(([section, key, kind, why, a, b, c]) => {
    const value = saved[key];
    const made = controlFor(kind, value, a, b, c);
    teardown.inputs[key] = { read: made.read, was: value === undefined ? null : value };
    return configRow(key, key, made, why);
  });
  const note = el("span", { class: "muted" }, teardown.message);
  return [
    el("h2", {}, "When a host is given up"),
    el("div", { class: "panel" },
      el("p", { class: "muted" }, "Every one of these decides how long a host that is not working still bills."),
      el("table", {}, el("tbody", {}, rows)),
      el("div", { class: "row" },
        el("button", { class: "primary", onclick: (e) => saveTeardown(e, note) }, "Save to configuration"),
        el("button", { class: "small", onclick: () => render() }, "Reset"),
        note)),
  ];
}

async function saveTeardown(event, note) {
  const body = { teardown: {} };
  for (const [key, { read, was }] of Object.entries(teardown.inputs)) {
    const now = read();
    if (JSON.stringify(now) !== JSON.stringify(was)) body.teardown[key] = now;
  }
  if (!Object.keys(body.teardown).length) { teardown.message = " nothing changed"; note.textContent = teardown.message; return; }
  const button = event.target;
  button.disabled = true;
  try {
    let answer;
    try {
      answer = await api.setSearch(body);
    } catch (error) {
      const changes = error.changes || [];
      const retype = changes.find((c) => c.requires_retype);
      if (!retype) throw error;
      const ok = await confirmAction({
        title: "This loosens a limit",
        body: el("div", {}, ...changes.map((c) => el("p", {}, c.detail))),
        retype: retype.value,
      });
      if (!ok) return;
      answer = await api.setSearch({ ...body, confirm: retype.value });
    }
    teardown.message = ` saved as version ${String(answer.version).slice(0, 8)}`;
  } catch (error) {
    teardown.message = ` ${error.message}`;
  } finally {
    note.textContent = teardown.message;
    button.disabled = false;
    setTimeout(render, 600);
  }
}
const allocation = { inputs: {}, mode: null, message: "" };

// Which listings the pool searches by itself (D80). Not a filter — a pool on `interruptible`
// never asks the on-demand listing, so a fixed-price host is not rejected, it is never seen.
// That was invisible here: the rejection list only names offers that were looked at.
const MODES = [
  ["interruptible", "interruptible — bid, and accept being outbid"],
  ["on_demand", "on demand — fixed price, cannot be outbid"],
  ["cheaper", "cheaper — search both listings, take whichever costs less"],
];

function searchSection(market) {
  const saved = (market.saved || {});
  search.saved = saved;
  search.inputs = {};
  search.mode = saved.mode || "interruptible";
  search.profile = saved.search_profile || "";
  search.saveAs = "";
  const profileSelect = el("select", { onchange: (e) => {
    const chosen = e.target.value;
    run(e.target, async () => { await api.setSearch({ search_profile: chosen }); render(); });
  } },
    el("option", { value: "", ...(search.profile ? {} : { selected: true }) },
      "(the pool's own offer_policy)"),
    ...(saved.search_profiles || []).map((name) =>
      el("option", { value: name, ...(search.profile === name ? { selected: true } : {}) }, name)));
  const modeSelect = el("select", {}, ...MODES.map(([value, label]) =>
    el("option", { value, ...(search.mode === value ? { selected: true } : {}) }, label)));
  const rows = SEARCH_FIELDS.filter(([section]) => saved[section]).map(([section, key, kind, why, a, b, c]) => {
    const value = saved[section][key];
    const made = controlFor(kind, value, a, b, c);
    search.inputs[`${section}.${key}`] = { read: made.read, section, key, was: value };
    return configRow(key, key, made, why);
  });

  const note = el("span", { class: "muted" }, search.message);
  if (!rows.length) {
    return [
      el("h2", {}, "What the pool looks for"),
      el("p", { class: "muted" }, "This pool's supervisor does not report its offer policy yet; restart it to edit the search here. Until then the Configuration screen is the place."),
    ];
  }
  return [
    el("h2", {}, "What the pool looks for"),
    el("div", { class: "panel" },
      el("div", { class: "row" },
        el("label", {}, "Search profile"), profileSelect,
        el("input", { type: "text", placeholder: "save these as…", style: "width:11rem",
                      oninput: (e) => { search.saveAs = e.target.value.trim(); } }),
        el("span", { class: "muted" },
          "a named set of these filters (D87) — save the values below under a name, and pick it here later")),
      el("div", { class: "row" },
        el("label", {}, "Rent by"), modeSelect,
        el("span", { class: "muted" },
          "which listings are searched at all — an offer in a listing the pool does not ask for "
          + "is never seen, and so never appears among the rejections below")),
      el("table", {}, el("tbody", {}, rows)),
      el("div", { class: "row" },
        // Deliberately not through `run`: that refreshes the screen, which would rebuild this
        // form from the saved values and throw away what was just typed into it.
        el("button", { class: "primary", onclick: async (e) => {
          const button = e.target, label = button.textContent;
          button.disabled = true; button.textContent = "asking…";
          try {
            const fresh = await api.market(1, searchValues());
            marketView.box.replaceChildren(marketPanel(fresh));
            marketView.at = Date.now();
            stampMarket();
            search.message = fresh.problem
              ? ` the market could not be asked: ${fresh.problem}`
              : ` tried, not saved — ${fresh.passed} of ${fresh.seen} offers pass these values`;
          } catch (error) {
            search.message = ` ${error.message}`;
          } finally {
            note.textContent = search.message;
            button.disabled = false; button.textContent = label;
          }
        } }, "Try these"),
        el("button", { onclick: (e) => saveSearch(e, note, modeSelect) }, "Save to configuration"),
        el("button", { class: "small", onclick: () => render() }, "Reset"),
        note)),
  ];
}

function allocationSection(market) {
  const saved = market.saved || {};
  allocation.inputs = {};
  if (!saved.dynamic) {
    return [
      el("h2", {}, "How capacity is decided"),
      el("p", { class: "muted" }, "This pool's supervisor does not report its allocation settings yet; restart it to edit them here."),
    ];
  }
  allocation.mode = saved.allocation || "lease";
  const mode = el("select", {},
    el("option", { value: "lease", ...(allocation.mode === "lease" ? { selected: true } : {}) },
      "lease — the lease's worker count is the demand"),
    el("option", { value: "dynamic", ...(allocation.mode === "dynamic" ? { selected: true } : {}) },
      "dynamic — the traffic is the demand, the lease is the ceiling"),
  );

  const rows = ALLOCATION_FIELDS.filter(([section]) => saved[section]).map(([section, key, kind, why, a, b, c]) => {
    const value = saved[section][key];
    const made = controlFor(kind, value, a, b, c);
    allocation.inputs[`${section}.${key}`] = { read: made.read, section, key, was: value };
    return configRow(key, `${section === "dynamic" ? "" : "workers_auto: "}${key}`, made, why);
  });

  const note = el("span", { class: "muted" }, allocation.message);
  return [
    el("h2", {}, "How capacity is decided"),
    el("div", { class: "panel" },
      el("div", { class: "row" },
        el("label", {}, "Allocation"), mode,
        el("span", { class: "muted" },
          "Nothing is rented without an open lease and its dollar cap either way.")),
      el("table", {}, el("tbody", {}, rows)),
      el("div", { class: "row" },
        el("button", { class: "primary", onclick: (e) => saveAllocation(e, note, mode) },
          "Save to configuration"),
        el("button", { class: "small", onclick: () => render() }, "Reset"),
        note)),
  ];
}

function allocationValues() {
  const body = { allocation: null, dynamic: {}, workers_auto: {} };
  for (const { read, section, key } of Object.values(allocation.inputs)) body[section][key] = read();
  return body;
}

async function saveAllocation(event, note, mode) {
  const button = event.target, label = button.textContent;
  const wanted = allocationValues();
  const body = { dynamic: {}, workers_auto: {} };
  for (const { section, key, was } of Object.values(allocation.inputs)) {
    const now = wanted[section][key];
    if (JSON.stringify(now) !== JSON.stringify(was ?? null)) body[section][key] = now;
  }
  if (mode.value !== allocation.mode) body.allocation = mode.value;
  if (!body.allocation && !Object.keys(body.dynamic).length && !Object.keys(body.workers_auto).length) {
    allocation.message = " nothing changed";
    note.textContent = allocation.message;
    return;
  }
  button.disabled = true; button.textContent = "saving…";
  try {
    let answer;
    try {
      answer = await api.setSearch(body);
    } catch (error) {
      // Switching allocation on can loosen a limit; it comes back refused, with the plan.
      const changes = error.changes || [];
      const retype = changes.find((c) => c.requires_retype);
      if (!retype) throw error;
      const ok = await confirmAction({
        title: "This loosens a limit",
        body: el("div", {}, ...changes.map((c) => el("p", {}, c.detail))),
        retype: retype.value,
      });
      if (!ok) return;
      answer = await api.setSearch({ ...body, confirm: retype.value });
    }
    allocation.message = ` saved · ${(answer.changes || []).length} change(s) applied`;
    note.textContent = allocation.message;
    await refresh();
  } catch (error) {
    allocation.message = ` ${error.message}`;
    note.textContent = allocation.message;
  } finally {
    button.disabled = false; button.textContent = label;
  }
}

function searchValues() {
  const body = { offer_policy: {}, bidding: {} };
  for (const { read, section, key } of Object.values(search.inputs)) body[section][key] = read();
  return body;
}

function changedValues(mode) {
  const body = { offer_policy: {}, bidding: {} };
  const wanted = searchValues();
  for (const { section, key, was } of Object.values(search.inputs)) {
    const now = wanted[section][key];
    if (JSON.stringify(now) !== JSON.stringify(was ?? null)) body[section][key] = now;
  }
  // `mode` sits directly under `rented`, beside the policy rather than inside it.
  if (mode && mode.value !== search.mode) body.mode = mode.value;
  // A name typed into "save these as" sends the whole form as a new profile, not as an edit
  // to the one in force: saving is how you branch away from what you are looking at.
  if (search.saveAs) { body.save_profile_as = search.saveAs; body.offer_policy = searchValues().offer_policy; }
  return body;
}

async function saveSearch(event, note, mode) {
  const body = changedValues(mode);
  const count = Object.values(body).reduce(
    (n, section) => n + (typeof section === "object" ? Object.keys(section).length : 1), 0);
  if (!count) { search.message = " nothing changed"; note.textContent = search.message; return; }
  const button = event.target;
  button.disabled = true;
  try {
    let answer;
    try {
      answer = await api.setSearch(body);
    } catch (error) {
      // A change that loosens a limit comes back refused, with the plan that says why.
      const changes = error.changes || [];
      const retype = changes.find((c) => c.requires_retype);
      if (!retype) throw error;
      const ok = await confirmAction({
        title: "This loosens a limit",
        body: el("div", {}, ...changes.map((c) => el("p", {}, c.detail))),
        retype: retype.value,
      });
      if (!ok) return;
      answer = await api.setSearch({ ...body, confirm: retype.value });
    }
    search.message = ` saved · ${(answer.changes || []).length} change(s) applied`;
    note.textContent = search.message;
    await refresh();
  } catch (error) {
    alert(error.message);
  } finally {
    button.disabled = false;
  }
}

const MARKET_REFRESH_S = 60;
const marketView = { box: null, stamp: null, button: null, busy: false, at: 0 };

function marketSection(initial) {
  marketView.box = el("div", {}, marketPanel(initial));
  marketView.stamp = el("span", { class: "muted" });
  marketView.button = el("button", { class: "small", onclick: () => refreshMarket() }, "Refresh");
  marketView.at = Date.now();
  stampMarket();
  return [
    el("div", { class: "row" },
      el("h2", {}, "Live market — the real offer pipeline, read-only"), marketView.button, marketView.stamp),
    marketView.box,
  ];
}

const marketOnScreen = () =>
  state.screen === "rented" && marketView.box !== null && document.body.contains(marketView.box);

function stampMarket() {
  if (!marketView.stamp) return;
  const next = Math.max(0, Math.round(MARKET_REFRESH_S - (Date.now() - marketView.at) / 1000));
  marketView.stamp.textContent = marketView.busy
    ? " asking the provider…"
    : ` updated ${new Date(marketView.at).toLocaleTimeString()} · refreshes in ${next}s`;
}

async function refreshMarket() {
  if (marketView.busy || !marketOnScreen()) return;
  marketView.busy = true;
  marketView.button.disabled = true;
  stampMarket();
  try {
    const fresh = await api.market(1).catch((error) => ({ error: error.message }));
    if (marketOnScreen()) {  // the operator may have moved on while it was asked
      marketView.box.replaceChildren(marketPanel(fresh));
    }
  } finally {
    marketView.at = Date.now();
    marketView.busy = false;
    marketView.button.disabled = false;
    stampMarket();
  }
}

// One timer for the life of the page, not one per visit to the screen.
setInterval(() => {
  if (!marketOnScreen()) return;
  if (Date.now() - marketView.at >= MARKET_REFRESH_S * 1000) refreshMarket();
  else stampMarket();
}, 1000);

// The number of hosts the pool may rent at once. A configuration change like any other, so it
// goes validate → plan → apply, and raising it must be typed again (spec §1): more hosts is
// more money. The pool rents up to this many as demand needs them — it is a ceiling, not an
// order to rent that many.
function hostLimitControl(current) {
  const input = el("input", { type: "number", min: "0", max: "50", step: "1", value: String(current), style: "width:4.5rem" });
  const set = async (e) => {
    const wanted = Number(input.value);
    if (!Number.isInteger(wanted) || wanted < 0 || wanted === current) return;
    const button = e.target; button.disabled = true;
    try {
      const { text, version } = await api.getConfig();
      const pattern = /max_rented_hosts:\s*\d+/g;
      const found = text.match(pattern) || [];
      if (found.length !== 1) {
        alert("Could not find a single max_rented_hosts in the configuration file; change it on the Configuration screen.");
        return;
      }
      const candidate = text.replace(pattern, `max_rented_hosts: ${wanted}`);
      const plan = await api.planConfig(candidate);
      if (plan.errors && plan.errors.length) { alert(plan.errors.join("\n")); return; }
      const body = el("div", {}, ...plan.changes.map((c) => el("p", {}, c.detail)));
      const ok = await confirmAction({
        title: `Rent up to ${wanted} host${wanted === 1 ? "" : "s"} at once?`,
        body,
        retype: plan.changes.some((c) => c.requires_retype) ? String(wanted) : null,
      });
      if (!ok) return;
      await api.applyConfig(candidate, version);
      await refresh();
    } catch (error) {
      alert(error.message);
    } finally { button.disabled = false; }
  };
  return el("div", { class: "row" }, input, el("button", { class: "small", onclick: set }, "Set"));
}

const marketPanel = (market) => {
  if (market.error) return el("p", { class: "error" }, market.error);
  const skipped = Object.entries(market.avoided || {});
  const avoidedNote = skipped.length ? el("p", { class: "muted" },
    "Skipped for now because they just failed: ",
    skipped.map(([machine, why]) => `${machine} (${why})`).join(" · ")) : null;
  // "Could not ask" is not "nothing out there" — say which, or an operator reads a throttled
  // provider as an empty market and goes looking for the wrong problem (D44).
  if (market.problem) {
    return el("div", { class: "panel" },
      el("p", { class: "error" }, "The market could not be asked, so this is not a picture of what is out there."),
      el("p", { class: "mono" }, market.problem),
      el("p", { class: "muted" }, "Nothing will be rented until this clears. A provider that is rate-limiting usually just needs fewer passes: raise probe_interval_s, or close leases you are not using."));
  }
  return el("div", {}, avoidedNote, el("div", { class: "grid" },
    el("div", { class: "panel" },
      el("div", { class: "stat" }, `${market.passed} pass · ${market.rejected} rejected`),
      el("div", { class: "muted" }, `${market.seen} offers seen through your policy`),
      el("h2", {}, "Rejected, by reason"),
      el("table", {}, el("tbody", {}, Object.entries(market.rejected_by_reason || {}).map(([reason, count]) =>
        el("tr", {}, el("td", { class: "num" }, count), el("td", { class: "muted" }, reason)))))),
    el("div", { class: "panel" }, el("h2", {}, "Best offers"),
      el("table", {},
        el("thead", {}, el("tr", {},
          el("th", {}, "Hardware"), el("th", {}, "Kind"), el("th", { class: "num" }, "Floor"), el("th", { class: "num" }, "Would pay"),
          el("th", { class: "num" }, "On-demand"), el("th", { class: "num" }, "$/GB"),
          el("th", { class: "num", title: "Workers a host rented from this offer would run" }, "Workers"),
          el("th", { class: "num" }, "Score"), el("th", {}, ""))),
        el("tbody", {}, (market.best || []).map((offer, index) => el("tr", {},
          el("td", {}, index === 0 ? el("strong", {}, offer.hardware) : offer.hardware,
            el("div", { class: "muted mono" }, `${offer.machine} · ${offer.gpu_memory_gb}GB · ${offer.download_mbps}Mbps`)),
          // A bid can be outbid at any moment; a fixed price cannot. Same machine, different deal.
          el("td", {}, offer.kind === "on_demand"
            ? pill("on demand", "ok") : pill("bid", "warn")),
          el("td", { class: "num" }, offer.kind === "on_demand" ? "—" : rate(offer.floor)),
          el("td", { class: "num" }, el("strong", {}, rate(offer.would_bid))),
          el("td", { class: "num" }, rate(offer.on_demand)),
          el("td", { class: "num" }, `$${Number(offer.download_per_gb).toFixed(4)}`),
          // A profile's number is shown plainly; the default is marked, so it reads as unmeasured.
          el("td", { class: "num", title: offer.workers_from || "" },
            offer.workers ?? "—",
            offer.workers_from && offer.workers_from.startsWith("capacity profile") ? "" : el("span", { class: "muted" }, " default")),
          el("td", { class: "num" }, Math.round(offer.score)),
          el("td", {}, offer.offer_id
            ? el("button", { class: "small", onclick: (e) => rentThis(e, offer) }, "Rent")
            : null))))))));
};

// Renting one particular offer, the way it is listed: this machine, as a bid or at its fixed
// price. It is still the pool's policy deciding what may be rented — the list only shows what
// passes — and if the offer has gone by the time it is asked for, nothing is rented instead.
async function rentThis(event, offer) {
  const spend = Number(document.getElementById("prepare-spend")?.value || 1);
  const hours = Number(document.getElementById("prepare-hours")?.value || 1);
  const when = document.getElementById("prepare-when")?.value || "join";
  const fixed = offer.kind === "on_demand";
  const ok = await confirmAction({
    title: `Rent ${offer.hardware}?`,
    body: el("div", {},
      el("p", {}, fixed
        ? `On demand at ${rate(offer.would_bid)} — a fixed price. Nobody can outbid it; it runs until the lease ends or you release it.`
        : `A bid of ${rate(offer.would_bid)} (floor ${rate(offer.floor)}). Cheaper, and it can be outbid at any moment.`),
      el("p", {}, `Worst case: ${money(spend)} over ${hours}h, this one host. Machine ${offer.machine} · ${offer.gpu_memory_gb} GB · ${offer.workers} workers.`),
      el("p", { class: "muted" }, "Cap and time limit come from the Prepare a host panel above. If this offer has gone, nothing else is rented in its place.")),
  });
  if (!ok) return;
  run(event.target, () => api.prepare({
    max_spend: spend, max_hours: hours, when_ready: when, offer_id: offer.offer_id, kind: offer.kind,
  }));
}

function preparePanel() {
  const spend = el("input", { id: "prepare-spend", type: "number", step: "0.01", value: "1.00", style: "width:6rem" });
  const hours = el("input", { id: "prepare-hours", type: "number", step: "0.5", value: "1", style: "width:5rem" });
  const when = el("select", { id: "prepare-when" }, el("option", { value: "join" }, "join the pool"), el("option", { value: "park" }, "park it"), el("option", { value: "destroy" }, "destroy"));
  const kind = el("select", {},
    el("option", { value: "" }, "as the pool is configured"),
    el("option", { value: "interruptible" }, "bid — cheaper, can be outbid"),
    el("option", { value: "on_demand" }, "on demand — fixed price, cannot be outbid"));
  return el("div", { class: "panel" }, el("h2", {}, "Prepare a host"),
    el("p", { class: "muted" }, "Its own small lease: rent, load the model set, verify, then join, park or destroy. Borrows no authority from any other lease."),
    el("label", {}, "Dollar cap ", spend),
    el("label", {}, "Time limit (hours) ", hours),
    el("label", {}, "When ready ", when),
    el("label", {}, "Rent it ", kind),
    el("p", { class: "muted" }, "Or pick one machine: every row in the market below has its own Rent button."),
    el("div", { class: "row" }, el("button", { class: "primary", onclick: async (e) => {
      const worst = el("div", {}, el("p", {}, `This spends money. Worst case: ${money(Number(spend.value))} over ${hours.value}h, one host.`),
        el("p", { class: "muted" }, "The pool bids on the best offer its policy allows, and stops at the cap less the safety margin."));
      if (!(await confirmAction({ title: "Prepare a host?", body: worst }))) return;
      run(e.target, () => api.prepare({
        max_spend: Number(spend.value), max_hours: Number(hours.value), when_ready: when.value,
        ...(kind.value ? { kind: kind.value } : {}),
      }));
    } }, "Prepare a host")));
}

screens.models = (status) => {
  // What the pool resolved for each (host, logical model) — the build that is actually served
  // there, and whether *that tag* is resident. A logical name is never what the engine holds.
  const rows = [];
  for (const host of [...status.hosts, ...status.rented]) {
    for (const model of status.model_set || []) {
      const served = (host.served || {})[model];
      rows.push({ host: host.host_id, model, served, residency: host.residency || "pinned" });
    }
  }
  // Loaded is what a pinned host needs; on disk is enough for an on-demand one, whose engine
  // loads on first use. Not on disk is missing on either: nothing is ever downloaded for a request.
  const residentPill = (row) => {
    const label = variant(row.model, row.served.tag);
    if (row.served.resident) return pill("resident ✓" + label, "ok");
    if (row.served.available && row.residency === "on_demand") return pill("on disk · loads on use" + label, "ok");
    if (row.served.available) return pill("on disk, not loaded ✗" + label, "bad");
    return pill("missing ✗" + label, "bad");
  };
  const schema = (value) => value === true ? "enforces" : value === false ? "ignores" : "unknown";
  // "mlx variant" for gemma4:e4b served as gemma4:e4b-mlx; nothing when the tag is the name.
  const variant = (model, tag) => {
    if (!tag || tag === model) return "";
    const suffix = tag.startsWith(model) ? tag.slice(model.length).replace(/^[-:_]/, "") : tag;
    return ` (${suffix} variant)`;
  };
  return [
    el("h1", {}, "Models"),
    el("p", { class: "muted" }, "Every host holds the pool's whole model set, loaded, all the time. A host that does not is not routed to."),
    el("div", { class: "panel" }, el("h2", {}, "The pool's model set"),
      el("div", { class: "mono" }, (status.model_set || []).join("  ·  ") || "—")),
    el("h2", {}, "Per host"),
    el("table", {},
      el("thead", {}, el("tr", {}, el("th", {}, "Host"), el("th", {}, "Model"), el("th", {}, "Build served"),
        el("th", {}, "Runtime class"), el("th", {}, "Schema"), el("th", {}, "Resident"))),
      el("tbody", {}, rows.map((row) => el("tr", {},
        el("td", { class: "mono" }, row.host),
        el("td", { class: "mono" }, row.model),
        el("td", { class: "mono" }, row.served ? row.served.tag : el("span", { class: "muted" }, "no usable variant")),
        el("td", { class: "muted" }, row.served ? row.served.runtime_class : "—"),
        el("td", { class: "muted" }, row.served ? schema(row.served.enforces_schema) : "—"),
        el("td", {}, row.served ? residentPill(row) : pill("not served", "bad")))))),
  ];
};

screens.leases = async () => {
  const { leases } = await api.leases();
  const workers = el("input", { type: "number", value: "2", min: "1", style: "width:5rem" });
  const hours = el("input", { type: "number", value: "1", step: "0.5", style: "width:5rem" });
  const spend = el("input", { type: "number", value: "", step: "0.01", placeholder: "required", style: "width:7rem" });
  const allow = el("input", { type: "checkbox" });
  return [
    el("h1", {}, "Leases"),
    el("div", { class: "panel" }, el("h2", {}, "Open a lease"),
      el("p", { class: "muted" }, "A lease is the only thing that can spend. One that may rent must state its dollars."),
      el("div", { class: "row" },
        el("label", {}, "Workers ", workers),
        el("label", {}, "Max hours ", hours),
        el("label", {}, "Max spend ", spend),
        el("label", {}, "May rent ", allow),
        el("button", { class: "primary", onclick: async (e) => {
          const body = { workers: Number(workers.value), max_hours: Number(hours.value), allow_rent: allow.checked };
          if (spend.value !== "") body.max_spend = Number(spend.value);
          if (allow.checked) {
            const worst = el("div", {}, el("p", {}, `This authorises spending. Worst case: ${money(body.max_spend)} over ${body.max_hours}h.`));
            if (!(await confirmAction({ title: "Open a lease that can rent?", body: worst }))) return;
          }
          run(e.target, () => api.openLease(body));
        } }, "Open lease"))),
    ...leaseTables(leases),
  ];
};

// Open leases first and always; the closed ones behind a filter (D86). A pool that has run for
// a day has dozens of closed leases, and the one that matters — the one spending money now —
// was at the bottom of an unfiltered list.
const leaseFilter = { show: "closed-recent", text: "" };

function leaseTables(leases) {
  const open = leases.filter((l) => l.state === "open");
  const closed = leases.filter((l) => l.state !== "open");

  const matches = (l) => {
    if (leaseFilter.text && !JSON.stringify(l).toLowerCase().includes(leaseFilter.text.toLowerCase())) return false;
    if (leaseFilter.show === "closed-none") return false;
    if (leaseFilter.show === "closed-spent") return (l.spent_enforced_on || 0) > 0;
    return true;
  };
  const showing = closed.filter(matches).slice(0, leaseFilter.show === "closed-all" ? 500 : 10);

  const search = el("input", {
    type: "search", placeholder: "filter by id or reason…", value: leaseFilter.text,
    style: "width:16rem", oninput: (e) => { leaseFilter.text = e.target.value; render(); },
  });
  const which = el("select", { onchange: (e) => { leaseFilter.show = e.target.value; render(); } },
    ...[["closed-recent", "the last 10"], ["closed-spent", "only those that spent"],
        ["closed-all", "all of them"], ["closed-none", "none"]].map(([v, label]) =>
      el("option", { value: v, ...(leaseFilter.show === v ? { selected: true } : {}) }, label)));

  return [
    el("h2", {}, `Open (${open.length})`),
    open.length ? leaseRows(open) : el("p", { class: "muted" }, "No open lease, so nothing can be rented."),
    el("h2", {}, `Closed (${closed.length})`),
    el("div", { class: "row" }, el("label", {}, "Show "), which, search,
      el("span", { class: "muted" }, `${showing.length} shown`)),
    showing.length ? leaseRows(showing) : el("p", { class: "muted" }, "Nothing matches."),
  ];
}

const leaseRows = (leases) => el("table", {},
  el("thead", {}, el("tr", {},
    el("th", {}, "Lease"), el("th", {}, "State"), el("th", { class: "num" }, "Workers"),
    el("th", { class: "num" }, "Spent (enforced)"), el("th", { class: "num" }, "Estimated"),
    el("th", { class: "num" }, "Cap"), el("th", { class: "num" }, "Left"), el("th", {}, ""))),
  el("tbody", {}, leases.map((lease) => el("tr", {},
    el("td", { class: "mono" }, lease.lease_id),
    el("td", {}, pill(lease.state, lease.state === "open" ? "ok" : ""), lease.closed_reason ? el("div", { class: "muted" }, lease.closed_reason) : null),
    el("td", { class: "num" }, lease.workers),
    el("td", { class: "num" }, money(lease.spent_enforced_on)),
    el("td", { class: "num" }, money(lease.estimated_spend)),
    el("td", { class: "num" }, money(lease.max_spend)),
    el("td", { class: "num" }, money(lease.dollars_left)),
    el("td", {}, leaseActions(lease))))));

screens.decisions = async () => {
  const { events } = await api.events(200);
  return [
    el("h1", {}, "Decisions"),
    el("p", { class: "muted" }, "Every decision the pool made, with the numbers that produced it. Click a row to expand."),
    el("table", {},
      el("thead", {}, el("tr", {}, el("th", {}, "When"), el("th", {}, "Kind"), el("th", {}, "Host"), el("th", {}, "Summary"))),
      el("tbody", {}, events.flatMap((event) => {
        const detail = el("tr", { class: "detail", hidden: true },
          el("td", { colspan: "4" }, el("pre", {}, JSON.stringify(event.numbers, null, 2))));
        const row = el("tr", { class: "expandable", onclick: () => { detail.hidden = !detail.hidden; } },
          el("td", { class: "muted mono" }, clock(event.ts)),
          el("td", {}, pill(event.kind)),
          el("td", { class: "mono muted" }, event.host_id || "—"),
          el("td", {}, event.summary));
        return [row, detail];
      }))),
  ];
};

screens.config = async () => {
  const current = await api.getConfig();
  state.config = current;
  const editor = el("textarea", { class: "mono", spellcheck: "false" }, current.text);
  const output = el("div", {});
  const applyButton = el("button", { class: "primary", disabled: true }, "Apply");
  let plannedText = null;

  const showPlan = async (button) => {
    const label = button.textContent; button.disabled = true; button.textContent = "planning…";
    try {
      const plan = await api.planConfig(editor.value);
      plannedText = editor.value;
      applyButton.disabled = plan.errors.length > 0;
      output.replaceChildren(
        plan.errors.length
          ? el("div", { class: "panel" }, el("h2", { class: "error" }, "Will not load"), el("pre", { class: "error" }, plan.errors.join("\n")))
          : el("div", { class: "panel" },
              el("h2", {}, plan.changes.length ? "What this change would cause, now" : "No change"),
              plan.changes.length
                ? el("ul", {}, plan.changes.map((change) => el("li", {},
                    change.detail,
                    change.requires_retype ? el("span", { class: "pill warn" }, " must be retyped ") : null,
                    change.needs_restart ? el("span", { class: "pill warn" }, " needs a router restart ") : null)))
                : el("p", { class: "muted" }, "The file parses to the same configuration."))
      );
    } catch (error) {
      output.replaceChildren(el("p", { class: "error" }, error.message));
    } finally { button.disabled = false; button.textContent = label; }
  };

  applyButton.onclick = async (e) => {
    const plan = await api.planConfig(editor.value);
    const loosening = plan.changes.filter((c) => c.requires_retype);
    if (loosening.length) {
      const ok = await confirmAction({
        title: "This loosens a limit",
        body: el("div", {}, loosening.map((c) => el("p", {}, c.detail))),
        retype: loosening[0].value,
      });
      if (!ok) return;
    }
    run(e.target, async () => {
      await api.applyConfig(editor.value, current.version);
      await render();
    });
  };

  const history = await api.configHistory().catch(() => ({ versions: [] }));
  return [
    el("h1", {}, "Configuration"),
    el("p", { class: "muted" }, "The file is the one source of truth. Nothing is applied blind: validate, then plan, then apply. Changing configuration never spends money."),
    el("div", { class: "row" }, el("span", { class: "muted mono" }, `version ${current.version.slice(0, 12)} · ${current.path}`)),
    editor,
    el("div", { class: "row" },
      el("button", { onclick: async (e) => {
        const button = e.target; const label = button.textContent; button.disabled = true;
        try {
          const result = await api.validateConfig(editor.value);
          output.replaceChildren(result.ok
            ? el("p", { class: "ok" }, "Valid.")
            : el("pre", { class: "error" }, result.errors.join("\n")));
        } finally { button.disabled = false; button.textContent = label; }
      } }, "Validate"),
      el("button", { onclick: (e) => showPlan(e.target) }, "Plan"),
      applyButton),
    output,
    el("h2", {}, "History"),
    (history.versions || []).length
      ? el("table", {},
          el("thead", {}, el("tr", {}, el("th", {}, "Applied"), el("th", {}, "Version"), el("th", {}, ""))),
          el("tbody", {}, history.versions.map((version) => el("tr", {},
            el("td", { class: "muted" }, new Date(version.applied_at * 1000).toLocaleString()),
            el("td", { class: "mono" }, version.version.slice(0, 12)),
            el("td", {}, el("button", { class: "small", onclick: (e) => run(e.target, async () => {
              await api.rollbackConfig(version.version); await render();
            }) }, "Roll back"))))))
      : el("p", { class: "muted" }, "No applied versions yet."),
  ];
};

// --- shell ---

async function render() {
  const name = location.hash.replace("#", "") || "overview";
  state.screen = name;
  for (const link of document.querySelectorAll("#nav a")) {
    link.classList.toggle("active", link.getAttribute("href") === `#${name}`);
  }
  const main = document.getElementById("screen");
  try {
    const pending = (screens[name] || screens.overview)(state.status);
    if (pending instanceof Promise) {
      // A screen that has to ask the provider takes seconds. Say so, rather than leaving the
      // previous screen on the page where it reads as this one's answer.
      main.replaceChildren(el("p", { class: "muted" }, "Loading…"));
    }
    const parts = await pending;
    if (state.screen !== name) return;  // the operator moved on while we were waiting
    main.replaceChildren(...[parts].flat().filter(Boolean));
  } catch (error) {
    if (state.screen !== name) return;
    main.replaceChildren(el("p", { class: "error" }, error.message));
  }
}

async function refresh() {
  state.status = await api.status();
  document.getElementById("pool-name").textContent = state.status.pool;
  await render();
}

async function stream() {
  // fetch, not EventSource: the admin key travels as a header, never in the URL.
  while (true) {
    try {
      const response = await fetch(`/pool/events/stream?after=${state.lastEventId}`, {
        headers: { Authorization: `Bearer ${ADMIN_KEY}` },
      });
      if (!response.ok) throw new Error(`stream: ${response.status}`);
      document.getElementById("link-state").replaceWith(pillState("live", "ok"));
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const chunks = buffer.split("\n\n");
        buffer = chunks.pop();
        for (const chunk of chunks) handleFrame(chunk);
      }
    } catch {
      document.getElementById("link-state").replaceWith(pillState("reconnecting", "warn"));
      await new Promise((resolve) => setTimeout(resolve, 3000));
    }
  }
}

function pillState(text, kind) {
  const node = pill(text, kind);
  node.id = "link-state";
  return node;
}

function handleFrame(chunk) {
  const lines = chunk.split("\n");
  const type = lines.find((l) => l.startsWith("event: "))?.slice(7);
  const data = lines.filter((l) => l.startsWith("data: ")).map((l) => l.slice(6)).join("");
  if (!data) return;
  const payload = JSON.parse(data);
  if (type === "decision") {
    state.lastEventId = Math.max(state.lastEventId, payload.id);
    state.events.unshift(payload);
    state.events = state.events.slice(0, 200);
    if (state.screen === "overview") render();
  } else if (type === "status") {
    state.status = payload;
    document.getElementById("pool-name").textContent = payload.pool;
    if (["overview", "hosts", "models"].includes(state.screen)) render();
  }
}

async function start() {
  const { events } = await api.events(50);
  state.events = events;
  state.lastEventId = events.length ? Math.max(...events.map((e) => e.id)) : 0;
  await refresh();
  stream();
}

window.addEventListener("hashchange", render);
document.getElementById("panic").onclick = async (e) => {
  const ok = await confirmAction({
    title: "Release all rented hosts?",
    body: "Every rented host is destroyed now, verified, and every open lease is closed. Local and fixed hosts keep serving.",
  });
  if (ok) run(e.target, () => api.down());
};

const keyDialog = document.getElementById("key-dialog");
document.getElementById("key-form").onsubmit = async (event) => {
  event.preventDefault();
  ADMIN_KEY = document.getElementById("key-input").value.trim();
  const error = document.getElementById("key-error");
  try {
    await api.status();
    keyDialog.close();
    await start();
  } catch (problem) {
    ADMIN_KEY = null;
    error.textContent = problem.message;
    error.hidden = false;
  }
};
keyDialog.showModal();
