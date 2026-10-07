"use strict";
/* The GPM console.
 *
 * One more client of the control API: every button here is a call the `pool` CLI can also
 * make. No framework and no build step — the whole page is this file, index.html and
 * console.css, served by the supervisor.
 *
 * The admin key goes out as an Authorization header on each call, never in a URL or a cookie.
 * It is kept for this tab (sessionStorage: a reload keeps it, closing the tab forgets it), or —
 * when the operator ticks "Keep me signed in" — for this browser until Sign out (localStorage).
 * The page's Content-Security-Policy lets no script run here but this file (D137).
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
  // The server searches only when asked to, by name (D120, D122); without it, the settings.
  // What to search narrows by the market's chips: which providers, which kinds (providers.md §7.2).
  market: (hours = 4, policy, search = true) => {
    const narrow = `&kinds=${marketView.kinds}` + (marketView.connections
      ? `&connections=${[...marketView.connections].map(encodeURIComponent).join(",")}` : "");
    return policy
      ? call("POST", `/pool/market/preview?hours=${hours}${narrow}&search=true`, policy)
      : call("GET", `/pool/market/preview?hours=${hours}${narrow}&search=${search ? "true" : "false"}`);
  },
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
  modelSize: (repo) => call("GET", `/pool/models/size?${new URLSearchParams({ repo })}`),
  searchModels: (q) => call("GET", `/pool/models/search?${new URLSearchParams({ q })}`),
  refreshDirectory: () => call("POST", "/pool/directory/refresh"),
  setHostService: (id, disabled) => call("PATCH", `/pool/config/hosts/${encodeURIComponent(id)}`, { disabled }),
  removeHost: (id, confirm) => call("DELETE", `/pool/config/hosts/${encodeURIComponent(id)}`, confirm === undefined ? {} : { confirm }),
  restartEngine: (hostId, applySettings) =>
    call("POST", `/pool/hosts/${hostId}/engine/restart`, { confirm: hostId, apply_settings: applySettings }),
  deleteModel: (hostId, tag) => call("POST", `/pool/hosts/${hostId}/models/delete`, { tag, confirm: tag }),
  workloads: () => call("GET", "/pool/workloads"),
  planWorkload: (body) => call("POST", "/pool/workloads/plan", body),
  createWorkload: (body) => call("POST", "/pool/workloads", body),
  extendWorkload: (name, body) => call("POST", `/pool/workloads/${encodeURIComponent(name)}/extend`, body),
  endWorkload: (name) => call("POST", `/pool/workloads/${encodeURIComponent(name)}/end`),
  rotateWorkloadKey: (name) => call("POST", `/pool/workloads/${encodeURIComponent(name)}/keys`),
  provisioners: () => call("GET", "/pool/provisioners"),
  createProvisioner: (body) => call("POST", "/pool/provisioners", body),
  revokeProvisioner: (name, end) => call("DELETE", `/pool/provisioners/${encodeURIComponent(name)}?end_workloads=${end ? "true" : "false"}`),
  // Provider accounts (D130, D134-D136). A credential goes out once, in a body, and never comes back.
  providers: (fresh = false) => call("GET", `/pool/providers${fresh ? "?fresh=true" : ""}`),
  plugin: (type) => call("GET", `/pool/providers/plugins/${encodeURIComponent(type)}`),
  testProvider: (body) => call("POST", "/pool/providers/test", body),
  addProvider: (body) => call("POST", "/pool/providers", body),
  changeProvider: (name, body) => call("PATCH", `/pool/providers/${encodeURIComponent(name)}`, body),
  removeProvider: (name, confirm) => call("DELETE", `/pool/providers/${encodeURIComponent(name)}`, confirm === undefined ? {} : { confirm }),
  setCredential: (name, credential) => call("PUT", `/pool/providers/${encodeURIComponent(name)}/credential`, { credential }),
  removeCredential: (name) => call("DELETE", `/pool/providers/${encodeURIComponent(name)}/credential`),
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
    facts.push(["rented as", (PRICED[pricedOf(d)] || PRICED.bid)[2]
      + (d.connection ? ` · through ${d.connection}` : "")]);
    facts.push(["cost", `${rate(d.bid_hourly)} · held ${((d.hours_held || 0) * 60).toFixed(0)} min · spent ${money(d.estimated_spend)} (provider says ${money(d.reported_spend)})`]);
    facts.push(["lease", d.lease_id || "—"]);
  }
  if (d.provider) facts.push(["the provider says", `${d.provider.state}${d.provider.detail ? " · " + d.provider.detail : ""}`]);
  if (d.tunnel) facts.push(["tunnel", `${d.tunnel.up ? "up" : "down"} on :${d.tunnel.local_port} · ${d.tunnel.restarts} restart(s)`]);
  facts.push(["engine", engine.answers
    ? `answers · ${(engine.on_disk || []).length} on disk, ${(engine.loaded || []).length} loaded`
    // Silent on purpose (D105): an engine started once its weights have landed is not a fault
    // while the agent fetches them, and was read as one when it said "ReadError" here.
    : engine.expected ? el("span", { class: "muted" }, engine.detail || "not started yet")
    : el("span", { class: "error" }, `not answering: ${engine.detail || "?"}`)]);

  // One row per model the host must hold, with its state from everything the pool knows
  // (D105): downloading with its bar, on disk, loading, loaded, failed with the reason.
  const MODEL_PILL = { loaded: "ok", loading: "warn", "on disk": "warn", downloading: "warn", failed: "bad", "not here yet": "bad" };
  const modelRows = (d.models || []).map((m) => el("tr", {},
    el("td", { class: "mono" }, m.tag),
    el("td", {}, pill(m.state, MODEL_PILL[m.state] || "warn")),
    el("td", {}, m.detail || "",
      m.state === "downloading" && m.total
        ? el("div", { class: "burn" }, el("div", { style: `width:${(Math.min(1, m.completed / m.total) * 100).toFixed(1)}%` }))
        : null)));

  return [
    el("div", { class: "kv" }, ...facts.flatMap(([k, v]) => [el("div", { class: "k" }, k), el("div", {}, v)])),
    modelRows.length ? el("div", {},
      el("h2", {}, "The model set on this host"),
      el("table", {}, el("tbody", {}, modelRows))) : null,
    // From a supervisor that does not report per-model states yet: the old view, from the
    // engine's own lists and the pool's downloads.
    !d.models && progress.length ? el("div", {},
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
    engine.answers && !d.models ? el("div", {},
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

function confirmAction({ title, body, retype, retypeLabel, okLabel }) {
  const dialog = document.getElementById("confirm-dialog");
  document.getElementById("confirm-title").textContent = title;
  const holder = document.getElementById("confirm-body");
  holder.replaceChildren(typeof body === "string" ? el("p", {}, body) : body);
  const retypeBox = document.getElementById("confirm-retype");
  const input = document.getElementById("confirm-input");
  const ok = document.getElementById("confirm-ok");
  document.getElementById("confirm-retype-label").textContent = retypeLabel || "Type the value again to confirm";
  ok.textContent = okLabel || "Confirm";
  retypeBox.hidden = !retype;
  input.value = "";
  // Loosening is harder than tightening: the new value must be typed again (spec §1). A number
  // matches however it is written: "$0.57", "0.57" and ".57" are the same budget.
  const same = (typed) => {
    const bare = typed.trim().replace(/^\$/, "");
    const want = String(retype);
    return /^\d*\.?\d+$/.test(bare) && /^\d*\.?\d+$/.test(want) ? Number(bare) === Number(want) : bare === want;
  };
  const check = () => { ok.disabled = retype ? !same(input.value) : false; };
  check();
  input.oninput = check;
  // A dialog keeps the answer it last closed with, and Escape does not replace it: without this,
  // Escape after any confirmed dialog confirmed the next one, retype and all.
  dialog.returnValue = "";
  dialog.showModal();
  if (retype) input.focus();
  return new Promise((resolve) => {
    dialog.addEventListener("close", () => resolve(dialog.returnValue === "ok" && (!retype || same(input.value))), { once: true });
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
  el("p", { class: "muted" }, "Hosts you configured, rather than rented. Take one out of service or remove it here; adding and editing one are on the Configuration screen. Rented hosts are released from Rented capacity."),
  el("table", {},
    el("thead", {}, el("tr", {},
      el("th", {}, "Host"), el("th", {}, "Kind"), el("th", {}, "Engine"), el("th", {}, "Transport"), el("th", {}, "State"),
      el("th", { class: "num" }, "Workers"), el("th", {}, "Capabilities"), el("th", {}, "Residency"), el("th", {}, "Tunnel"), el("th", { class: "num" }, "Served"), el("th", {}, ""))),
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
      el("td", { class: "num" }, host.requests_served ?? 0),
      el("td", {}, hostControls(host)))))),
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

// Rented capacity is five screens' worth of settings; one long page hid which belonged to
// what. Each tab answers one question an operator comes here with, and has its own address so
// Back works and a tab can be linked to.
const RENTED_TABS = [
  { id: "providers", label: "Providers", about: "The provider accounts this pool rents from: each one's credential, what it can do, what it holds, and adding another." },
  { id: "hosts", label: "Hosts", about: "What this pool is renting now, what it costs, and renting one more by hand." },
  { id: "engine", label: "Profiles", about: "What each machine this pool rents holds — its models, each with the variant it fetches — and the engine, image and start they run with." },
  { id: "finding", label: "Finding machines", about: "What the pool looks for on the market, and the live market those settings let through — try values here before saving them." },
  { id: "scaling", label: "Scaling", about: "How many machines at most, how much per hour at most, and how the pool decides when to rent another." },
  { id: "teardown", label: "Tear-down", about: "When an unused or failing machine is paused, destroyed or given up." },
];
// Which tabs read the saved settings. Opening one never searches the market: the provider
// counts every offer a search returns against a daily quota (D120), so the console searches
// only when the operator presses "Search the market" or "Try these".
const RENTED_TABS_WITH_MARKET = new Set(["finding", "scaling", "teardown"]);

function rentedTabs(current) {
  return el("div", { class: "tabs" }, ...RENTED_TABS.map((tab) => el("a", {
    href: `#rented/${tab.id}`, ...(tab.id === current ? { class: "active" } : {}),
  }, tab.label)));
}

screens.rented = async (status) => {
  if (!status.provider) return [el("h1", {}, "Rented capacity"), el("p", { class: "muted" }, "This pool has no rented capacity configured, so it cannot spend.")];
  // With no tab named it opens on Hosts — what is being paid for now.
  const tab = RENTED_TABS.find((t) => t.id === (state.sub || "hosts")) || RENTED_TABS[1];
  const market = RENTED_TABS_WITH_MARKET.has(tab.id) ? await api.market(1, null, false).catch((e) => ({ error: e.message })) : null;
  const head = [el("h1", {}, "Rented capacity"), rentedTabs(tab.id), el("p", { class: "muted tab-about" }, tab.about)];
  if (tab.id === "providers") return [...head, ...await providersTab()];
  if (tab.id === "engine") return [...head, ...engineSection(status)];
  if (tab.id === "finding") return [...head, ...searchSection(market), ...marketSection(market)];
  if (tab.id === "scaling") return [...head, limitsPanel(status), ...allocationSection(market)];
  if (tab.id === "teardown") return [...head, ...teardownSection(market)];
  return [...head, ...await rentedHostsTab(status)];
};

// Today's use of the provider's daily search quota (D121): counted from the pool's own searches,
// and the provider's own figure once it has refused.
function searchQuotaText(quota) {
  if (!quota) return null;
  const resets = new Date(quota.resets_at * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const share = quota.limit ? quota.used / quota.limit : 0;
  return el("span", { class: quota.exhausted ? "error" : share >= 0.8 ? "warn-text" : "muted" },
    quota.exhausted
      ? `search quota used up (${quota.limit.toLocaleString()} offers today) — every search is refused until ${resets}`
      : `search quota: ${quota.used.toLocaleString()} of ${quota.limit.toLocaleString()} offers used today (${Math.round(share * 100)}%) · resets at ${resets}`);
}

function providerPanel(status, account) {
  const capabilities = Object.entries(status.provider.capabilities).filter(([, on]) => on).map(([name]) => name);
  return el("div", { class: "panel" }, el("h2", {}, "Provider"),
    el("div", { class: "kv" },
      el("div", { class: "k" }, "name"), el("div", {}, status.provider.name),
      el("div", { class: "k" }, "credential"), el("div", {}, account.error ? pill("error", "bad") : pill(account.credential_valid ? "valid ✓" : "invalid ✗", account.credential_valid ? "ok" : "bad")),
      el("div", { class: "k" }, "credit left"), el("div", {}, account.error ? account.error : money(account.credit_remaining)),
      el("div", { class: "k" }, "can"), el("div", { class: "muted" }, capabilities.join(", ")),
      ...(status.provider.search_quota ? [
        el("div", { class: "k" }, "search quota"),
        el("div", {}, searchQuotaText(status.provider.search_quota),
          el("div", { class: "muted" }, "Offers returned by this pool's searches today. Renting and replacing hosts need it; other tools using the same account key are counted only once the provider refuses.")),
      ] : []),
      el("div", { class: "k" }, "cap margin"), el("div", {}, `${(status.provider.cap_safety_margin * 100).toFixed(0)}%`,
        el("span", { class: "muted" }, status.provider.capabilities.reports_charges ? " — narrows once a charge is reported" : " — wider: this provider reports no charges"))),
    el("p", { class: "muted" }, (status.providers || []).length > 1
      ? `The first of ${status.providers.length} provider accounts. `
      : "", el("a", { href: "#rented/providers" }, "Provider accounts →")));
}

// --- Provider accounts (D130, D134-D136; providers.md §7.1) ---
//
// A card per account, every card laid out alike so two read side by side: who it is, whether it
// is working, its credential, what it holds, what it can and cannot do, and what can be done to
// it. A credential is typed into a dialog, sent once, and dropped from the page as the dialog
// closes.

const providersView = { tests: {}, data: null, secrets: [], focus: null, opened: 0 };

// Hues far enough apart that two providers' lettermarks never read as one.
const LETTERMARK_HUES = [215, 160, 275, 25, 340, 190, 45, 120];

// A plug-in's own icon is drawn as an image — never as markup, so it can run nothing. Without
// one, a lettermark in a colour fixed by the provider's name.
function providerIcon(p, size = 40) {
  const name = p.display_name || p.type || "?";
  if (p.icon) {
    return el("img", { class: "provider-icon", width: size, height: size, alt: "",
      src: `data:image/svg+xml;charset=utf-8,${encodeURIComponent(p.icon)}` });
  }
  if (p.icon_url && p.icon_url.startsWith("https://")) {
    // The provider's own logo, from its site: no referrer sent, and the lettermark if it fails.
    const img = el("img", { class: "provider-icon", width: size, height: size, alt: "", src: p.icon_url,
      referrerpolicy: "no-referrer", loading: "lazy", decoding: "async" });
    img.addEventListener("error", () => img.replaceWith(providerIcon({ ...p, icon_url: null }, size)), { once: true });
    return img;
  }
  // "Lambda Cloud" → LC, "RunPod" → RP, "Vast.ai" → V.
  const words = name.split(/\s+/).filter(Boolean);
  const capitals = name.replace(/[^A-Z]/g, "");
  const letters = words.length > 1 ? words[0][0] + words[1][0] : capitals.length > 1 ? capitals.slice(0, 2) : name[0];
  let turn = 0;
  for (const c of String(p.type || name)) turn = (turn * 31 + c.charCodeAt(0)) % 997;
  const hue = LETTERMARK_HUES[turn % LETTERMARK_HUES.length];
  return el("div", { class: "provider-icon lettermark", "aria-hidden": "true",
    style: `--hue:${hue};width:${size}px;height:${size}px;font-size:${Math.round(size * (letters.length > 1 ? 0.38 : 0.48))}px` },
  letters.toUpperCase());
}

const providerLabel = (p) => (p.display_name && p.display_name !== p.connection ? `${p.display_name} (${p.connection})` : p.connection);

// Small line icons for what a provider can do; drawn here, never fetched.
const CAP_ICON = {
  interruptible: '<path d="M4 12h4l2-6 4 12 2-6h4"/>',
  parkable: '<rect x="7" y="6" width="3" height="12" rx="1"/><rect x="14" y="6" width="3" height="12" rx="1"/>',
  self_terminate: '<circle cx="12" cy="13" r="7"/><path d="M12 13V9M10 3h4"/>',
  interruption_notice: '<path d="M12 4l9 16H3z"/><path d="M12 10v4M12 17v.5"/>',
  reports_charges: '<path d="M6 3h12v18l-3-2-3 2-3-2-3 2z"/><path d="M9 8h6M9 12h6"/>',
  volumes: '<ellipse cx="12" cy="6" rx="7" ry="3"/><path d="M5 6v12c0 1.7 3.1 3 7 3s7-1.3 7-3V6"/>',
  copies: '<rect x="8" y="8" width="11" height="11" rx="2"/><path d="M5 15V5h10"/>',
  reports_instance_logs: '<path d="M5 5h14M5 10h14M5 15h9M5 20h6"/>',
  warning: '<path d="M12 4l9 16H3z"/><path d="M12 10v4M12 17v.5"/>',
};
const capIcon = (name) => el("span", { class: "cap-icon", "aria-hidden": "true",
  html: `<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${CAP_ICON[name] || ""}</svg>` });

// What a provider can do, in the order an operator weighs it: how it rents, then what keeps
// spending safe, then what makes a host ready sooner. Every provider rents on demand.
const CAPABILITIES = [
  ["interruptible", "Interruptible", "Offers that can be taken away — bid for, or at the provider's spot price. Cheaper."],
  ["parkable", "Park", "A host can be stopped with its disk kept, and started again without downloading."],
  ["interruption_notice", "Interruption warning", "The provider warns before it takes a host back; the pool drains it at once."],
  ["self_terminate", "Dead-man timer", "A host ends itself if the pool goes quiet. Without it, leases here are kept short."],
  ["reports_charges", "Reports charges", "The provider says what each host cost, and spend is reconciled against it."],
  ["volumes", "Keeps models", "Storage that outlives a host, so a later host can take its models from it instead of downloading them."],
  ["copies", "Copies between hosts", "A new host takes its models from a sibling instead of the hub."],
  ["reports_instance_logs", "Boot logs", "A host that never answered says why."],
];

// Two rows, said in words: what it can do, and what it cannot — never told apart by style alone.
function capabilityRows(capabilities, { compact = false } = {}) {
  const has = CAPABILITIES.filter(([key]) => (capabilities || {})[key]);
  const lacks = CAPABILITIES.filter(([key]) => !(capabilities || {})[key]);
  const chip = ([key, label, about], on) => {
    const warn = !on && key === "self_terminate";
    // How far kept storage reaches is what decides whether it helps (D139).
    const reach = key === "volumes" && on ? (capabilities.volume_reach === "data_center" ? " — in a data center" : " — on one machine") : "";
    return el("li", { class: `cap-chip${on ? "" : " off"}${warn ? " warn" : ""}`, title: about },
      capIcon(warn ? "warning" : key), warn ? "No dead-man timer" : label + reach);
  };
  const row = (title, list, on) => el("div", { class: "cap-row" },
    el("span", { class: "cap-row-title" }, title),
    list.length ? el("ul", { class: "cap-chips", "aria-label": title }, list.map((c) => chip(c, on)))
      : el("span", { class: "muted small-text" }, on ? "none declared" : "nothing missing"));
  return el("div", { class: "cap-rows" }, row("Can", has, true), compact ? null : row("Cannot", lacks, false));
}

// The one thing to know first about an account: a state, with what it means for what it holds.
function providerHealth(p) {
  const holds = p.holds > 0 ? ` — ${p.holds} host${p.holds === 1 ? "" : "s"} held there` : "";
  const account = p.account || {};
  const credential = p.credential || {};
  if (!p.configured) return { key: "removed", tone: "parked", label: `Removed — still releasing${holds}` };
  if (credential.source === "missing") return { key: "credential", tone: "bad", label: `Needs a credential${holds}` };
  if (credential.source === "withheld") return { key: "credential", tone: "warn", label: `Credential waits for a restart${holds}` };
  if (account.state === "refused") return { key: "credential", tone: "bad", label: `Credential refused${holds}` };
  if (account.state === "unreachable") return { key: "unreachable", tone: "warn", label: "Cannot be reached" };
  if (!p.enabled) return { key: "off", tone: "disabled", label: `Off${holds}` };
  if (p.pending) return { key: "pending", tone: "warn", label: "Not renting — a change waits for a restart" };
  if (!account.state || account.state === "unknown") return { key: "unchecked", tone: "warn", label: "Not checked yet" };
  if (p.search_quota && p.search_quota.exhausted) return { key: "quota", tone: "warn", label: "Search quota used up" };
  if (p.search_error) return { key: "failing", tone: "warn", label: "Searches failing" };
  return { key: "searching", tone: "ok", label: "Searching" };
}

function credentialLine(p) {
  const c = p.credential || {};
  const account = p.account || {};
  const when = (ts) => new Date(ts * 1000).toLocaleDateString([], { month: "short", day: "numeric" });
  const where = c.source === "stored" ? `saved ${when(c.set_at)}`
    : c.source === "environment" ? `from ${c.variable} in the supervisor's environment`
    : c.source === "plugin" ? "read by the plug-in itself"
    : c.source === "withheld" ? `${c.variable || "the environment's"} is not sent here until the supervisor restarts — or type one in`
    : c.variable ? `none — type one in, or set ${c.variable}` : "none — type one in";
  const verdict = c.source === "missing" || c.source === "withheld" ? null
    : account.state === "valid" ? pill("valid", "ok")
    : account.state === "refused" ? pill("refused", "bad")
    : account.state === "unreachable" ? pill("unreachable", "warn") : null;
  return el("div", {},
    verdict, verdict ? " " : null, el("span", {}, where),
    account.state === "valid" && account.credit != null ? el("span", { class: "muted" }, ` · ${dollars(account.credit)} credit`) : null,
    c.cleared_at ? el("div", { class: "warn-text" }, "The credential saved earlier was deleted: the account's endpoint changed after it was saved.") : null,
    c.source !== "missing" && account.state !== "valid" && account.detail ? el("div", { class: "muted" }, account.detail) : null);
}

function quotaMeter(quota) {
  if (!quota) return el("span", { class: "muted" }, "no daily search quota");
  const share = quota.limit ? Math.min(1, quota.used / quota.limit) : 0;
  return el("div", {},
    el("div", { class: `burn${quota.exhausted ? " bad" : share >= 0.8 ? " warn" : ""}`, role: "img",
      "aria-label": `${Math.round(share * 100)}% of today's search quota used` }, el("div", { style: `width:${Math.round(share * 100)}%` })),
    el("div", { class: "muted small-text" }, searchQuotaText(quota)));
}

function holdingLine(p) {
  const parts = [];
  if (p.rented) parts.push(`${p.rented} rented`);
  if (p.parked) parts.push(`${p.parked} parked`);
  if (p.volumes) parts.push(`${p.volumes} volume${p.volumes === 1 ? "" : "s"}`);
  if (p.held_back) parts.push(`${p.held_back} waiting for the account to answer after a restart`);
  return el("div", {}, parts.length ? parts.join(" · ") : el("span", { class: "muted" }, "nothing"),
    p.hourly ? el("span", { class: "muted" }, ` · ${rate(p.hourly)}`) : null);
}

function testResults(steps) {
  const mark = { passed: "✓", failed: "✗", warning: "!", skipped: "–" };
  const tone = { passed: "ok", failed: "bad", warning: "warn", skipped: "disabled" };
  const label = { credential: "Credential", search: "A search", instances: "This pool's hosts there", capabilities: "Capabilities" };
  return el("ol", { class: "steps" }, steps.map((s) => el("li", { class: `step ${s.status}` },
    el("span", { class: `pill ${tone[s.status] || ""}`, role: "img", "aria-label": s.status }, mark[s.status] || "?"),
    el("div", {}, el("strong", {}, label[s.step] || s.step), el("div", { class: "muted" }, s.detail)))));
}

function providerCard(p, data) {
  const health = providerHealth(p);
  const c = p.credential || {};
  const tested = providersView.tests[p.connection];
  const actions = [], aside = [], reasons = [];
  if (p.configured) {
    actions.push(el("button", { class: "small", onclick: (e) => testSaved(e.target, p) }, "Test connection"));
    if (c.can_type) {
      actions.push(el("button", { class: `small${c.source === "missing" ? " primary" : ""}`, disabled: !data.secure_transport || undefined,
        onclick: () => credentialDialog(p, data) }, c.source === "stored" ? "Replace credential" : "Type in a credential"));
      const orSet = c.variable ? "; or set " + c.variable + " in the supervisor's environment" : "";
      if (!data.secure_transport) reasons.push(`A credential can be typed in only over HTTPS or on the pool's own machine${orSet}.`);
    }
    actions.push(el("button", { class: "small", onclick: (e) => toggleProvider(e.target, p) }, p.enabled ? "Turn off" : "Turn on"));
    aside.push(el("button", { class: "small", onclick: () => editProviderDialog(p) }, "Settings"));
    aside.push(el("button", { class: "small danger", disabled: p.holds > 0 || undefined,
      onclick: (e) => removeProvider(e.target, p) }, "Remove"));
    if (p.holds > 0) reasons.push(`It holds ${p.holds} host${p.holds === 1 ? "" : "s"} or volume${p.holds === 1 ? "" : "s"}: release them before removing it.`);
  }
  return el("article", { class: `panel provider-card${p.enabled ? "" : " is-off"}`, "data-connection": p.connection,
    "aria-labelledby": `provider-${p.connection}` },
    el("header", { class: "provider-head" }, providerIcon(p),
      el("div", { class: "provider-title" },
        el("h2", { id: `provider-${p.connection}` }, p.display_name || p.type),
        el("div", { class: "muted mono" }, p.connection === p.type ? p.type : `${p.connection} · ${p.type}`)),
      el("span", { class: `pill status ${health.tone}`, role: "status" }, health.label)),
    el("div", { class: "kv provider-facts" },
      el("div", { class: "k" }, "credential"), credentialLine(p),
      el("div", { class: "k" }, "search quota"), quotaMeter(p.search_quota),
      el("div", { class: "k" }, "holding"), holdingLine(p),
      p.configured ? el("div", { class: "k" }, "models") : null,
      p.configured ? keepModelsControl(p) : null,
      p.search_error ? el("div", { class: "k" }, "last search") : null,
      p.search_error ? el("div", { class: "error" }, p.search_error) : null,
      p.pending ? el("div", { class: "k" }, "waiting") : null,
      p.pending ? el("div", { class: "warn-text" }, `The file's change ${p.pending}.`) : null),
    capabilityRows(p.capabilities),
    el("div", { class: "provider-actions" }, el("div", { class: "row" }, actions), el("div", { class: "row" }, aside)),
    reasons.length ? el("div", { class: "muted small-text" }, reasons.map((r) => el("div", {}, r))) : null,
    tested ? el("details", { class: "provider-test", open: tested.open || undefined,
      ontoggle: (e) => { tested.open = e.target.open; } },
      el("summary", {}, `Test at ${clock(tested.at)}: `, tested.steps.some((s) => s.status === "failed")
        ? el("span", { class: "error" }, "failed") : tested.steps.some((s) => s.status === "warning")
          ? el("span", { class: "warn-text" }, "passed with a warning") : el("span", { class: "ok-text" }, "passed")),
      testResults(tested.steps)) : null);
}

async function providersTab() {
  let data;
  try {
    data = await api.providers();
  } catch (error) {
    return [el("p", { class: "error" }, error.message), el("button", { onclick: () => render() }, "Try again")];
  }
  providersView.data = data;
  const live = data.connections.filter((p) => p.configured);
  const searching = live.filter((p) => providerHealth(p).key === "searching").length;
  const hosts = data.connections.reduce((n, p) => n + p.rented + p.parked, 0);
  const burn = data.connections.reduce((n, p) => n + (p.hourly || 0), 0);
  const offered = data.plugins.filter((p) => p.offered && !data.connections.some((c) => c.type === p.type));
  const add = el("button", { class: "primary", disabled: !offered.length || undefined, onclick: () => addProviderDialog(data) }, "Add provider");
  const focusAfter = providersView.focus;
  providersView.focus = null;
  if (focusAfter) {
    // Back to the card that was acted on, rather than the top of the page.
    setTimeout(() => document.querySelector(`.provider-card[data-connection="${CSS.escape(focusAfter)}"] button`)?.focus(), 0);
  }
  return [
    el("div", { class: "providers-summary panel" },
      el("div", { class: "summary-stats" },
        el("div", {}, el("div", { class: "stat" }, live.length), el("div", { class: "muted" }, `account${live.length === 1 ? "" : "s"}`)),
        el("div", {}, el("div", { class: "stat" }, searching), el("div", { class: "muted" }, "searching")),
        el("div", {}, el("div", { class: "stat" }, hosts), el("div", { class: "muted" }, "hosts held")),
        el("div", {}, el("div", { class: "stat" }, rate(burn)), el("div", { class: "muted" }, "burning"))),
      el("div", { class: "summary-actions" },
        el("div", { class: "row" },
          el("button", { onclick: (e) => run(e.target, async () => { providersView.tests = {}; await api.providers(true); }) }, "Check every account"),
          add),
        offered.length ? null : el("div", { class: "muted small-text" }, "Every installed provider is connected: a pool has one account per provider."))),
    data.secure_transport ? null : el("p", { class: "warn-text" },
      "This page is not on HTTPS or the pool's own machine, so a credential cannot be typed in here. Open the console over HTTPS, or from the pool's machine."),
    live.length || data.connections.length ? el("div", { class: "provider-grid" }, data.connections.map((p) => providerCard(p, data)))
      : el("div", { class: "panel" }, el("p", {}, "No provider account yet."), add.cloneNode(true)),
    el("p", { class: "muted" }, "A pool has one account per provider. Limits — hosts, hourly burn, leases — are the pool's, across every account. "
      + "Every search asks each account that is on, and every rental takes what is expected to cost least per worker-hour across them."),
  ];
}

async function testSaved(button, p) {
  providersView.focus = p.connection;
  await run(button, async () => {
    const answer = await api.testProvider({ connection: p.connection });
    providersView.tests[p.connection] = { at: Date.now() / 1000, steps: answer.steps, open: true };
    await api.providers(true);
  });
}

// A change the plan says must be typed again: asked, then sent again with the typed value.
async function withRetype(send, title) {
  try {
    return await send(undefined);
  } catch (error) {
    const loosening = (error.changes || []).find((c) => c.requires_retype);
    if (!loosening) throw error;
    const ok = await confirmAction({ title, retype: loosening.value, retypeLabel: `Type ${loosening.value} to confirm`,
      body: el("ul", {}, error.changes.map((c) => el("li", {}, c.detail))) });
    if (!ok) return null;
    return await send(loosening.value);
  }
}

// Keep models between hosts (D139): offered only where the provider's storage reaches a data
// center; anywhere else the box is greyed and says why, in the provider's own terms.
const KEEP_MODELS_WHY = {
  data_center: "Each workload renting here gets a model volume in its first host's data center. Later hosts there copy their models from it, each file checked, instead of downloading. Billed to the workload's lease; deleted when the workload ends.",
  machine: "This provider keeps a volume on one machine only. It would help only when that same machine is free again, which is rare, and it is billed the whole time.",
  none: "This provider keeps no storage between hosts.",
};

function keepModelsControl(p) {
  const reach = p.volume_reach || "none";
  const can = reach === "data_center";
  const why = KEEP_MODELS_WHY[reach] || KEEP_MODELS_WHY.none;
  const hint = `keep-models-${p.connection}`;
  const box = el("input", { type: "checkbox", checked: (can && p.keep_models) || undefined, disabled: !can || undefined,
    "aria-describedby": hint, onchange: (e) => setKeepModels(e.target, p) });
  return el("label", { class: `remember keep-models${can ? "" : " is-disabled"}`, title: why }, box,
    el("span", {}, "Keep models between hosts",
      p.model_volumes ? el("span", { class: "muted" }, ` · ${p.model_volumes} model volume${p.model_volumes === 1 ? "" : "s"} now`) : null),
    el("span", { class: "hint muted", id: hint }, can ? why : `Not available: ${why}`));
}

async function setKeepModels(box, p) {
  const on = box.checked;
  providersView.focus = p.connection;
  await run(box, async () => {
    const ok = await confirmAction({
      title: on ? `Keep models between hosts at ${providerLabel(p)}?` : `Stop keeping models at ${providerLabel(p)}?`,
      okLabel: on ? "Keep models" : "Stop keeping them",
      body: on
        ? "Each workload renting here gets a model volume in its first host's data center, sized to its models. It is billed to the workload's lease, whether or not a host has it, and deleted when the workload ends. Later hosts there copy their models from it instead of downloading; the data center is preferred only while that saves more than it costs."
        : "No new model volume is made here. Each one already made is deleted once no host has it, and later hosts download their models.",
    });
    if (!ok) { box.checked = !on; return; }
    await api.changeProvider(p.connection, { keep_models: on });
  });
}

async function toggleProvider(button, p) {
  providersView.focus = p.connection;
  delete providersView.tests[p.connection];
  await run(button, async () => {
    if (p.enabled) {
      const ok = await confirmAction({ title: `Turn off ${providerLabel(p)}?`, okLabel: "Turn off",
        body: "Nothing more is searched or rented there. Its hosts stay until they are released, and are still watched and charged." });
      if (!ok) return;
      await api.changeProvider(p.connection, { enabled: false });
    } else {
      await withRetype((confirm) => api.changeProvider(p.connection, { enabled: true, ...(confirm ? { confirm } : {}) }),
        `Turn on ${providerLabel(p)}?`);
    }
  });
}

async function removeProvider(button, p) {
  await run(button, async () => {
    try {
      await api.removeProvider(p.connection);
    } catch (error) {
      if (!error.changes) throw error;
      const ok = await confirmAction({ title: `Remove ${providerLabel(p)}?`, retype: p.connection, okLabel: "Remove",
        retypeLabel: `Type ${p.connection} to remove it`,
        body: el("div", {}, el("p", {}, "It leaves the pool's file, and its saved credential is deleted. The file keeps its own history."),
          el("ul", {}, error.changes.map((c) => el("li", {}, c.detail)))) });
      if (ok) {
        await api.removeProvider(p.connection, p.connection);
        delete providersView.tests[p.connection];
      }
    }
  });
}

// --- the dialog the add, credential and settings flows share ---

function providerDialog() {
  let dialog = document.getElementById("provider-dialog");
  if (!dialog) {
    dialog = el("dialog", { id: "provider-dialog", class: "provider-dialog", "aria-labelledby": "provider-dialog-title" });
    document.body.append(dialog);
    // Whatever was typed is dropped as the dialog closes, however it closes — the fields of
    // steps no longer on screen included.
    dialog.addEventListener("close", () => {
      for (const input of providersView.secrets) input.value = "";
      providersView.secrets = [];
      providersView.opened += 1;  // anything still in flight for this opening draws nothing
      dialog.replaceChildren();
    });
  }
  // Whatever an earlier opening still has in flight must not draw into this one.
  providersView.opened += 1;
  return { dialog, opening: providersView.opened };
}

const dialogTitle = (text) => el("h2", { id: "provider-dialog-title" }, text);

let secretCount = 0;
function secretField(label, hint) {
  const id = `secret-${++secretCount}`;
  const input = el("input", { id, type: "password", autocomplete: "off", spellcheck: "false", "data-secret": "",
    placeholder: "paste it here", "aria-describedby": `${id}-hint` });
  providersView.secrets.push(input);
  const toggle = el("button", { type: "button", class: "small", "aria-pressed": "false", "aria-label": "Show credential", onclick: () => {
    const shown = input.type === "password";
    input.type = shown ? "text" : "password";
    toggle.textContent = shown ? "Hide" : "Show";
    toggle.setAttribute("aria-pressed", String(shown));
  } }, "Show");
  return { input, node: el("div", { class: "field" }, el("label", { for: id }, label),
    el("div", { class: "row tight" }, input, toggle), el("small", { id: `${id}-hint`, class: "muted" }, hint)) };
}

const SECRET_HINT = "Sent once to this pool and kept by its supervisor in an owner-only file. It is never shown again — not here, not in a log.";

function credentialDialog(p, data) {
  const { dialog } = providerDialog();
  const secret = secretField(`${p.display_name || p.type} credential`, SECRET_HINT);
  const result = el("div", { role: "alert" });
  const save = el("button", { class: "primary", type: "submit", disabled: true }, "Test and save");
  secret.input.addEventListener("input", () => { save.disabled = !secret.input.value.trim(); });
  const form = el("form", { method: "dialog" });
  form.onsubmit = async (event) => {
    event.preventDefault();
    const value = secret.input.value.trim();
    if (!value) return;
    save.disabled = true;
    save.textContent = "testing…";
    try {
      await api.setCredential(p.connection, value);
      delete providersView.tests[p.connection];
      providersView.focus = p.connection;
      dialog.close();
      await refresh();
    } catch (error) {
      result.replaceChildren(el("p", { class: "error" }, error.message));
      save.disabled = false;
    } finally {
      save.textContent = "Test and save";
    }
  };
  const stored = (p.credential || {}).source === "stored";
  const deleteSaved = async () => {
    const ok = await confirmAction({ title: `Delete the saved credential for ${providerLabel(p)}?`, okLabel: "Delete",
      body: (p.credential || {}).variable
        ? `The account then takes ${p.credential.variable} from the supervisor's environment, if it is set.`
        : "The account then has no credential." });
    if (!ok) return;
    try {
      await api.removeCredential(p.connection);
      delete providersView.tests[p.connection];
      dialog.close();
      await refresh();
    } catch (error) { result.replaceChildren(el("p", { class: "error" }, error.message)); }
  };
  form.append(
    el("header", { class: "provider-head" }, providerIcon(p, 32), dialogTitle(`${stored ? "Replace" : "Type in"} the credential for ${providerLabel(p)}`)),
    p.holds ? el("p", { class: "muted" }, `The pool holds ${p.holds} host(s) or volume(s) here, so the new credential must see them — a key to another account is refused.`) : null,
    secret.node,
    result,
    stored && p.holds ? el("p", { class: "muted small-text" }, "The saved credential cannot be deleted while the account holds hosts: nothing could then watch or release them. Replace it instead.") : null,
    el("div", { class: "row" },
      el("button", { type: "button", onclick: () => dialog.close() }, "Cancel"),
      stored && !p.holds ? el("button", { type: "button", class: "danger", onclick: deleteSaved }, "Delete saved credential") : null,
      save));
  dialog.replaceChildren(form);
  dialog.showModal();
  secret.input.focus();
}

function editProviderDialog(p) {
  const { dialog } = providerDialog();
  const prior = el("input", { id: "provider-prior", type: "number", min: "0", step: "0.01", value: p.interruption_prior_per_hour ?? "",
    placeholder: "the pool's default", "aria-describedby": "provider-prior-hint" });
  const settings = el("textarea", { id: "provider-settings", class: "mono", rows: "5", spellcheck: "false", "aria-describedby": "provider-settings-hint" },
    JSON.stringify(p.settings || {}, null, 2));
  const note = el("div", { role: "alert" });
  const endpointKeys = p.endpoint_settings || [];
  const form = el("form", { method: "dialog" });
  form.onsubmit = async (event) => {
    event.preventDefault();
    let parsed;
    try { parsed = JSON.parse(settings.value || "{}"); } catch { parsed = undefined; }
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      note.replaceChildren(el("p", { class: "error" }, "Settings are a JSON object, like {} or {\"name\": \"value\"}."));
      return;
    }
    const moved = endpointKeys.filter((k) => JSON.stringify(parsed[k] ?? null) !== JSON.stringify((p.settings || {})[k] ?? null));
    if (moved.length && (p.credential || {}).source === "stored") {
      const ok = await confirmAction({ title: "Delete the saved credential?", okLabel: "Change and delete",
        body: `Changing ${moved.join(", ")} sends requests somewhere else, so when it takes effect — at the supervisor's next restart — the saved credential is deleted rather than sent there. Type it in again then.` });
      if (!ok) return;
    }
    const body = { settings: parsed, interruption_prior_per_hour: prior.value === "" ? null : Number(prior.value) };
    try {
      await api.changeProvider(p.connection, body);
      providersView.focus = p.connection;
      dialog.close();
      await refresh();
    } catch (error) {
      note.replaceChildren(el("p", { class: "error" }, error.message));
    }
  };
  form.append(
    el("header", { class: "provider-head" }, providerIcon(p, 32), dialogTitle(`Settings for ${providerLabel(p)}`)),
    el("div", { class: "field" }, el("label", { for: "provider-prior" }, "Interruptions expected per hour"), prior,
      el("small", { id: "provider-prior-hint", class: "muted" }, "For a machine with no history here, how often an interruptible host is taken away. Empty uses the pool's default.")),
    el("div", { class: "field" }, el("label", { for: "provider-settings" }, "Plug-in settings"), settings,
      el("small", { id: "provider-settings-hint", class: "muted" }, endpointKeys.length
        ? `${endpointKeys.join(", ")} decide where requests go: changing them deletes a saved credential. Never put a credential here. Takes effect when the supervisor restarts.`
        : "The plug-in's own settings. Never put a credential here. Takes effect when the supervisor restarts.")),
    note,
    el("div", { class: "row" }, el("button", { type: "button", onclick: () => dialog.close() }, "Cancel"),
      el("button", { class: "primary", type: "submit" }, "Save")));
  dialog.replaceChildren(form);
  dialog.showModal();
}

// Add provider: choose → name and credential → test → save, one step on screen at a time.
function addProviderDialog(data) {
  const { dialog, opening } = providerDialog();
  const current = () => providersView.opened === opening && dialog.open;
  const draft = { step: 1, plugin: null, name: "", nameEdited: false, enabled: true, tested: null, problem: null, plan: null, busy: false };
  const secret = secretField("Credential", SECRET_HINT);
  const STEPS = ["Provider", "Account", "Test", "Save"];

  const stepper = () => el("ol", { class: "stepper", "aria-label": "Steps" }, STEPS.map((label, i) =>
    el("li", { class: i + 1 === draft.step ? "current" : i + 1 < draft.step ? "done" : "",
      "aria-current": i + 1 === draft.step ? "step" : undefined }, el("span", { "aria-hidden": "true" }, i + 1), label)));
  const nav = (...buttons) => el("div", { class: "row wizard-nav" },
    el("button", { type: "button", onclick: () => dialog.close() }, "Cancel"),
    draft.step > 1 ? el("button", { type: "button", onclick: () => { draft.step -= 1; draft.problem = null; draw(); } }, "Back") : null,
    ...buttons);
  const problem = () => (draft.problem ? el("p", { class: "error", role: "alert" }, draft.problem) : null);

  function choose() {
    const connected = new Set(data.connections.map((c) => c.type));
    const offered = data.plugins.filter((p) => p.offered);
    const tile = (p) => {
      const pick = async () => {
        if (draft.busy) return;
        let chosen = p;
        if (!p.loaded) {
          draft.busy = p.type; draw();
          try { chosen = await api.plugin(p.type); } catch (error) { draft.problem = error.message; }
          draft.busy = false;
          if (!current()) return;
          if (draft.problem) { draw(); return; }
        }
        if (!draft.plugin || draft.plugin.type !== chosen.type) {
          // Another provider: nothing typed for the last one goes to this one.
          secret.input.value = "";
          if (!draft.nameEdited) draft.name = chosen.type;
          draft.tested = null; draft.plan = null;
        }
        draft.plugin = chosen; draft.problem = null; draft.step = 2; draw();
      };
      return el("button", { type: "button", class: `plugin-tile${draft.plugin && draft.plugin.type === p.type ? " chosen" : ""}`,
        disabled: draft.busy || undefined, onclick: pick },
        providerIcon(p, 36),
        el("div", {}, el("strong", {}, p.display_name),
          el("div", { class: "muted small-text" }, draft.busy === p.type ? "Loading…"
            : p.loaded ? p.type : `Separate package ${p.package}${p.version ? ` ${p.version}` : ""}. Choosing it loads its code into the pool.`),
          p.loaded ? capabilityRows(p.capabilities, { compact: true }) : null));
    };
    const free = offered.filter((p) => !connected.has(p.type));
    const taken = offered.filter((p) => connected.has(p.type));
    return [el("p", { class: "muted" }, "The installed provider plug-ins. A pool has one account per provider."),
      el("div", { class: "plugin-tiles" }, free.length ? free.map(tile) : el("p", { class: "muted" }, "No other provider plug-in is installed.")),
      taken.length ? el("p", { class: "muted small-text" }, `Already connected: ${taken.map((p) => p.display_name).join(", ")}.`) : null,
      problem(), nav()];
  }

  function account() {
    const p = draft.plugin;
    const name = el("input", { id: "provider-name", value: draft.name, autocomplete: "off", spellcheck: "false",
      pattern: "[a-z][a-z0-9_\\-]{0,31}", "aria-describedby": "provider-name-hint" });
    name.oninput = () => { draft.name = name.value.trim(); draft.nameEdited = true; };
    const form = el("form", { method: "dialog" });
    form.onsubmit = (event) => {
      event.preventDefault();
      if (!/^[a-z][a-z0-9_-]{0,31}$/.test(draft.name)) { draft.problem = "A lower-case name: letters, digits, - and _, starting with a letter."; draw(); return; }
      if (p.takes_credential && data.secure_transport && !secret.input.value.trim()) { draft.problem = "Paste the account's credential."; draw(); return; }
      draft.problem = null; draft.step = 3; draft.tested = null; draw(); test();
    };
    form.append(
      el("header", { class: "provider-head" }, providerIcon(p, 32), el("div", {}, el("strong", {}, p.display_name), capabilityRows(p.capabilities, { compact: true }))),
      el("div", { class: "field" }, el("label", { for: "provider-name" }, "Name in this pool"), name,
        el("small", { id: "provider-name-hint", class: "muted" }, "How hosts, spend and decisions name this account. It cannot be reused for another provider later.")),
      p.takes_credential
        ? (data.secure_transport ? secret.node : el("p", { class: "error" }, "Not on HTTPS or the pool's own machine: a credential cannot be sent from here. It can be set in the supervisor's environment instead."))
        : el("p", { class: "muted" }, "This plug-in reads its own credential from its settings or environment."),
      problem(), nav(el("button", { type: "submit", class: "primary" }, "Test it")));
    return [form];
  }

  async function test() {
    let answer;
    try {
      const value = secret.input.value.trim();
      answer = await api.testProvider({ type: draft.plugin.type, ...(value ? { credential: value } : {}) });
    } catch (error) {
      answer = { ok: false, steps: [{ step: "credential", status: "failed", detail: error.message }] };
    }
    if (!current() || draft.step !== 3) return;  // closed, or moved on, meanwhile
    draft.tested = answer;
    draw();
  }

  function testing() {
    if (!draft.tested) return [el("p", { class: "muted", role: "status" }, "Asking the provider…"), nav()];
    const warnings = draft.tested.steps.filter((s) => s.status === "warning");
    const understood = el("input", { id: "provider-understood", type: "checkbox" });
    const next = el("button", { type: "button", class: "primary", disabled: !draft.tested.ok || warnings.length > 0 || undefined,
      onclick: () => { if (warnings.length && !understood.checked) return; draft.step = 4; draft.plan = null; draw(); savePlan(); } }, "Continue");
    understood.onchange = () => { next.disabled = !understood.checked; };
    return [testResults(draft.tested.steps),
      draft.tested.ok && warnings.length ? el("label", { class: "pick option", for: "provider-understood" }, understood,
        `Add it anyway: ${warnings.map((w) => w.detail).join(" ")}`) : null,
      draft.tested.ok ? null : el("p", { class: "error", role: "alert" }, "Fix what failed, then go back and test again."),
      nav(el("button", { type: "button", onclick: () => { draft.tested = null; draw(); test(); } }, "Test again"), next)];
  }

  async function savePlan() {
    let plan;
    try {
      plan = (await api.addProvider({ name: draft.name, type: draft.plugin.type, enabled: draft.enabled, plan_only: true })).changes;
    } catch (error) {
      plan = [{ detail: error.message, refused: error.message }];
    }
    if (!current() || draft.step !== 4) return;
    draft.plan = plan;
    draw();
  }

  function saving() {
    const on = el("input", { id: "provider-on", type: "checkbox", checked: draft.enabled || undefined });
    on.onchange = () => { draft.enabled = on.checked; draft.plan = null; draw(); savePlan(); };
    const save = el("button", { type: "button", class: "primary", disabled: !draft.plan || draft.plan.some((c) => c.refused) || undefined }, "Add provider");
    save.onclick = async () => {
      save.disabled = true;
      const value = secret.input.value.trim();
      const body = { name: draft.name, type: draft.plugin.type, enabled: draft.enabled, ...(value ? { credential: value } : {}) };
      try {
        const done = await withRetype((confirm) => api.addProvider({ ...body, ...(confirm ? { confirm } : {}) }),
          `Add ${draft.plugin.display_name} (${draft.name}) and rent from it?`);
        if (done) {
          providersView.tests[draft.name] = { at: Date.now() / 1000, steps: draft.tested.steps };
          providersView.focus = draft.name;
          dialog.close();
          if (location.hash !== "#rented/providers") location.hash = "#rented/providers";
          else await refresh();
          return;
        }
      } catch (error) {
        draft.problem = error.message;
        draw();
        return;
      }
      save.disabled = false;
    };
    const credential = !draft.plugin.takes_credential ? "read by the plug-in itself"
      : secret.input.value.trim() ? "typed in — kept by the supervisor, never shown again" : "from the supervisor's environment";
    return [
      el("div", { class: "kv" },
        el("div", { class: "k" }, "provider"), el("div", {}, draft.plugin.display_name),
        el("div", { class: "k" }, "name"), el("div", { class: "mono" }, draft.name),
        el("div", { class: "k" }, "credential"), el("div", {}, credential)),
      el("label", { class: "pick option", for: "provider-on" }, on, "Search and rent from it now"),
      el("div", { class: "plan" }, el("strong", {}, "Saving it will:"),
        draft.plan ? el("ul", {}, draft.plan.map((c) => el("li", { class: c.refused ? "error" : "" }, c.refused || c.detail)))
          : el("p", { class: "muted", role: "status" }, "Working out what it changes…")),
      problem(), nav(save)];
  }

  function draw() {
    if (!current() && draft.step !== 1) return;
    const body = draft.step === 1 ? choose() : draft.step === 2 ? account() : draft.step === 3 ? testing() : saving();
    dialog.replaceChildren(dialogTitle("Add a provider account"), stepper(), ...body.filter(Boolean));
    if (draft.step === 2) (dialog.querySelector("input[data-secret]") || dialog.querySelector("#provider-name"))?.focus();
  }
  dialog.showModal();
  draw();
}

function limitsPanel(status) {
  return el("div", { class: "panel" }, el("h2", {}, "Limits"),
    el("div", { class: "kv" },
      el("div", { class: "k" }, "max rented hosts"), el("div", {}, hostLimitControl(status.limits.max_rented_hosts)),
      el("div", { class: "k" }, "per-host ceiling, all-in"), el("div", {}, rate(status.limits.per_host_ceiling)),
      el("div", { class: "k" }, "overall cap"), el("div", {}, status.limits.max_hourly_burn == null
        ? el("span", { class: "muted" }, `none — bounded at ${rate(status.limits.worst_case_hourly)} by hosts × ceiling`)
        : rate(status.limits.max_hourly_burn))),
    el("p", { class: "muted" }, "Raising either is a configuration change that must be retyped to confirm."));
}

async function rentedHostsTab(status) {
  const account = await api.account().catch((e) => ({ error: e.message }));
  // Read again on opening: the day's search quota (D121) moves with every search, and the
  // status the page holds may be from before the last one.
  status = await api.status().catch(() => status);
  const burn = status.rented.reduce((n, h) => n + (h.bid_hourly || 0), 0);
  return [
    el("div", { class: "grid" },
      providerPanel(status, account),
      el("div", { class: "panel" }, el("h2", {}, "Now"),
        el("div", { class: "kv" },
          el("div", { class: "k" }, "rented"), el("div", {}, `${status.rented.length} of at most ${status.limits.max_rented_hosts}`),
          el("div", { class: "k" }, "burning"), el("div", {}, rate(burn),
            el("span", { class: "muted" }, status.limits.max_hourly_burn == null ? "" : ` of ${rate(status.limits.max_hourly_burn)} allowed`)),
          el("div", { class: "k" }, "engine"), el("div", { class: "mono" }, status.engine?.rented || "—",
            el("a", { href: "#rented/engine", class: "muted" }, "  change")),
          el("div", { class: "k" }, "limits"), el("a", { href: "#rented/scaling" }, "on the Scaling tab"))),
      preparePanel()),
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
        el("td", {}, pricedPill(host), host.connection ? el("div", { class: "muted" }, host.connection) : null),
        el("td", { class: "num" }, rate(host.bid_hourly)),
        el("td", { class: "num" }, rate(host.storage_hourly)),
        el("td", { class: "num" }, `${(host.hours_held ?? 0).toFixed(2)}h`),
        el("td", { class: "num" }, money(host.estimated_spend), el("div", { class: "muted" }, `rep ${money(host.reported_spend)}`)),
        el("td", {}, el("div", { class: "row" },
          el("button", { class: "small", onclick: (e) => run(e.target, () => api.hostAction(host.host_id, "park")) }, "Park"),
          el("button", { class: "small danger", onclick: (e) => run(e.target, () => api.hostAction(host.host_id, "release")) }, "Release")))))))
      : el("p", { class: "muted" }, "Nothing rented."),
  ];
}

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
  ["offer_policy", "min_disk_gb", "range", "the disk each host is rented with — only machines offering this much are considered", 10, 500, 10],
  ["offer_policy", "max_all_in_hourly", "range", "the most this pool pays per host, per hour, all-in with that disk — bids and re-bids stop here", 0.05, 20, 0.05],
  ["offer_policy", "max_all_in_per_gpu", "range", "and per accelerator — a multi-GPU machine is judged by the card, not the bill", 0, 10, 0.05],
  ["offer_policy", "max_download_per_gb", "range", "the most it will pay per GB downloaded", 0, 0.5, 0.005],
  ["offer_policy", "min_download_mbps", "range", "slower than this and the model set takes too long", 0, 10000, 100],
  ["offer_policy", "min_reliability", "range", "the provider's own score, 0 to 1", 0, 1, 0.01],
  ["offer_policy", "min_driver_version", "text", "the accelerator driver this engine image needs — below it the card sits idle and the CPU serves"],
  ["offer_policy", "verified_only", "checkbox", "only machines the provider has verified"],
  ["offer_policy", "exclude_hardware", "list", "refused by name, case-insensitive"],
  ["offer_policy", "avoid_machines", "list", "machine ids to skip — one that keeps failing, say"],
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
  ["teardown", "avoid_lost_bid_minutes", "range", "a machine a bid just lost on is not bid on again for this long", 0, 240, 5],
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

// The same row with its explanation under the control rather than beside it — for editors
// whose controls are wide (a list of builds, a long select), where a third column is squeezed
// to a word per line.
function stackedRow(key, label, made, why) {
  return el("tr", {},
    el("td", { class: "mono stacked-label" }, label),
    el("td", {}, made.control, why ? el("div", { class: "muted why" }, why) : null));
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
  const engine = el("div", { class: "muted mono" }, host.engine || "",
    host.profile ? el("span", {}, ` · profile ${host.profile}`) : null);
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

function engineDraft(status) {
  const e = status.engine || {};
  const rented = e.rented || e.name || "ollama";
  const images = (e.images || []).filter((i) => i.min_driver);
  const made = profilesFromStatus(status, rented);
  return {
    rented,
    profiles: made.profiles,
    migrated: made.migrated,
    // Models and variants found by searching, new to the pool's file: {model: [{tag, engine, size_gb}]}.
    adds: {},
    // Which profile's finder is open, by index; one at a time.
    finding: null,
    images: images.length ? images.map((i) => ({ ...i })) : null,
    image: e.image || "",
    engine_start: e.engine_start || "",
    engine_options: [...(e.engine_options || [])],
    // What each engine's start can do, for the save to know whether a split may be sent.
    offers: e.offers || {},
  };
}

function engineSection(status) {
  if (!status.engine || !status.provider) return [];
  if (!engineEdit.draft) engineEdit.draft = engineDraft(status);
  const box = el("div", {});
  const draw = () => box.replaceChildren(...profilesTab(status, draw));
  draw();
  return [box];
}

// --- the Profiles tab (D111, D112) ---
//
// One place for what rented machines hold. A profile is a list of rows, each a model and the
// variant a machine fetches for it; every picker on this screen chooses the two together, from
// what the pool already has or from the hub and Ollama's library.

function profilesTab(status, draw) {
  const d = engineEdit.draft;
  const available = status.engine.available || ["ollama"];
  const engineSelect = el("select", { onchange: (ev) => { switchEngine(status, d, ev.target.value); draw(); } },
    ...available.map((name) => el("option", { value: name, ...(d.rented === name ? { selected: true } : {}) }, name)));
  const note = el("span", { class: "muted" }, engineEdit.message);
  return [
    el("h2", {}, "Profiles — what each rented machine holds"),
    el("div", { class: "panel" },
      el("div", { class: "row", style: "margin-top:0" },
        el("span", {}, "Rented machines run"), engineSelect,
        el("span", { class: "muted" }, "— every variant below is one this engine runs. Its image and start are under Machine setup.")),
      el("p", { class: "muted" },
        "Each machine the pool rents is bought as one ticked profile: one holding a model no host serves yet first, then the one whose requests are waiting most. "
        + "What a profile holds sets the least card and disk the search asks for."),
      d.migrated ? el("p", { class: "muted" },
        "This pool has no profiles yet: below is what it rents today, written as profiles. Saving keeps renting the same until you change them.") : null,
      ...d.profiles.map((profile, i) => profileCard(status, d, profile, i, draw)),
      el("div", { class: "row" },
        el("button", { onclick: () => {
          d.migrated = false;  // no longer what the pool rents today
          d.profiles.push({ name: `profile ${d.profiles.length + 1}`, rented: true, several: false, models: [] });
          d.finding = d.profiles.length - 1;
          finder.reset();
          draw();
        } }, "+ New profile"),
        d.profiles.some((p) => p.rented) ? null : el("span", { class: "error" }, "tick at least one profile to rent as"))),
    machineSetup(status, d, draw),
    el("div", { class: "row" },
      el("button", { class: "primary", onclick: (ev) => saveEngine(ev, note) }, "Save to configuration"),
      el("button", { class: "small", onclick: () => {
        engineEdit.draft = engineDraft(status); engineEdit.message = ""; finder.reset(); draw();
      } }, "Reset"),
      note),
    el("p", { class: "muted" }, "Saved as one change to the file, with its comments kept. Machines already rented keep what they were bought with; new ones use this."),
  ];
}

function switchEngine(status, d, engine) {
  d.rented = engine;
  d.migrated = false;
  // Only the options the chosen engine offers survive the switch; the file refuses the rest.
  const offered = ((status.engine.offers || {})[engine] || {}).options || {};
  d.engine_options = d.engine_options.filter((o) => o in offered);
  if (engine === "vllm") {
    d.images = d.images || VLLM_IMAGE_SUGGESTIONS.map((i) => ({ ...i }));
    if (/ollama/i.test(d.engine_start)) d.engine_start = "";
  }
  // A variant is one engine's: keep a row only where the pool has a variant of that model for the
  // new engine, and say which rows need choosing again.
  for (const profile of d.profiles) {
    for (const row of profile.models) {
      const variant = variantsFor(status, d).find((v) => v.model === row.model);
      Object.assign(row, variant ? { build: variant.tag, size_gb: variant.size_gb, precision: variant.precision } : { build: "", size_gb: null, precision: null });
    }
  }
  finder.results = null;
}

// Every variant the pool already has that rented machines can run, a model and its variant
// together — what "in the pool" offers, and what a profile's rows are checked against.
function variantsFor(status, d) {
  const caps = new Set(status.engine.rented_capabilities || []);
  const found = (status.engine.pool_variants || [])
    .filter((v) => v.engine === d.rented && (v.requires || []).every((c) => caps.has(c)));
  for (const [model, list] of Object.entries(d.adds)) {
    for (const v of list) if (v.engine === d.rented) found.push({ model, tag: v.tag, size_gb: v.size_gb, precision: v.precision || null, requires: [] });
  }
  return found;
}

function profileCard(status, d, profile, i, draw) {
  const finding = d.finding === i;
  const shape = (several, label) => el("label", { class: "pick" },
    el("input", { type: "radio", name: `shape-${i}`, ...(profile.several === several ? { checked: true } : {}),
      onchange: () => {
        profile.several = several;
        // One model per machine holds one: the first stays, and the card shows it.
        if (!several && profile.models.length > 1) profile.models = profile.models.slice(0, 1);
        draw();
      } }), el("span", {}, label));
  const rows = profile.models.map((m, at) => el("tr", {},
    el("td", { class: "mono" }, m.model),
    el("td", { class: "mono" }, m.build || el("span", { class: "error" }, `no ${d.rented} variant — choose one`)),
    el("td", {}, m.precision || "—"),
    el("td", { class: "num" }, m.size_gb ? `${m.size_gb} GB` : m.build ? el("span", { class: "muted" }, "not measured") : "—"),
    el("td", {}, el("button", { class: "small", title: "remove from this profile",
      onclick: () => { profile.models.splice(at, 1); draw(); } }, "×"))));
  const full = !profile.several && profile.models.length >= 1;
  return el("div", { class: "profile" + (finding ? " editing" : "") },
    el("div", { class: "row", style: "margin-top:0" },
      el("label", { class: "pick" }, el("input", { type: "checkbox", ...(profile.rented ? { checked: true } : {}),
        onchange: (ev) => { profile.rented = ev.target.checked; draw(); } }), el("span", {}, "rent")),
      el("input", { type: "text", class: "profile-name", value: profile.name, maxlength: "40", "aria-label": "profile name",
        oninput: (ev) => { profile.name = ev.target.value; } }),
      el("span", { class: "muted" }, "a machine holds"), shape(false, "one model"), shape(true, "several models"),
      el("span", { style: "margin-left:auto" }),
      el("button", { class: "small danger", onclick: () => {
        d.migrated = false;
        d.profiles.splice(i, 1);
        if (d.finding === i) d.finding = null; else if (d.finding > i) d.finding -= 1;
        draw();
      } }, "Remove profile")),
    profile.models.length ? el("table", { class: "builds held" },
      el("thead", {}, el("tr", {}, ...["Model", "Variant", "Precision", "Size", ""].map((h) => el("th", {}, h)))),
      el("tbody", {}, ...rows)) : el("p", { class: "muted" }, "Holds nothing yet — add a model below."),
    splitsAcross(status, d, profile, draw),
    el("div", { class: "row" }, needsLine(profile.models, profile.cards || 1),
      profile.several && d.rented === "vllm" && profile.models.length > 1
        ? el("span", { class: "muted" }, "· one vLLM process per model, the card's memory split between them") : null),
    finding
      ? finderView(status, d, profile, draw)
      : el("div", { class: "row" }, el("button", { class: "small", onclick: () => { d.finding = i; finder.reset(); draw(); } },
        full ? "Replace the model…" : "+ Add a model…")));
}

// A model too large for one card, split across a group of them (D114): offered only for an
// engine whose start can do it. The search then asks for whole groups of cards, each holding
// that share of every model.
function splitsAcross(status, d, profile, draw) {
  if (!(((status.engine.offers || {})[d.rented] || {}).splits_across_cards)) return null;
  const select = el("select", { "aria-label": "split each model", onchange: (ev) => { profile.cards = Number(ev.target.value); draw(); } },
    ...[1, 2, 4, 8].map((n) => el("option", { value: String(n), ...((profile.cards || 1) === n ? { selected: true } : {}) },
      n === 1 ? "no — a copy on every card" : `across ${n} cards`)));
  return el("div", { class: "row" },
    el("span", {}, "split each model"), select,
    (profile.cards || 1) > 1 ? el("span", { class: "muted" },
      `one copy per ${profile.cards} cards; a machine's cards must come in whole groups of ${profile.cards}`) : null);
}

// What a machine holding these needs, per card and on disk: the server's rule (sizing.py), which
// is the launcher's own. The Finding machines tab shows the number the search actually used.
const SIZING = { weightOverhead: 1.10, cacheReserveGb: 3 * 1024 ** 3 / 1e9, share: 0.90, diskOverhead: 1.10, diskHeadroomGb: 10 };

// Split across `cards` cards (D114), each card holds that share of every model's weights.
function needsFor(models, cards = 1) {
  const known = models.filter((m) => m.size_gb);
  const unknown = models.filter((m) => !m.size_gb).map((m) => m.model);
  if (!known.length) return { card: 0, disk: 0, weights: 0, unknown };
  const weights = known.reduce((sum, m) => sum + Number(m.size_gb), 0);
  return {
    card: Math.ceil((weights * SIZING.weightOverhead / cards + SIZING.cacheReserveGb * known.length) / SIZING.share),
    disk: Math.ceil(weights * SIZING.diskOverhead + SIZING.diskHeadroomGb),
    weights: Math.round(weights * 10) / 10,
    unknown,
  };
}

function needsLine(models, cards = 1) {
  if (!models.length) return null;
  const needs = needsFor(models, cards);
  const card = cards > 1 ? `cards in groups of ${cards}, each ≥ ${needs.card} GB,` : `a card of ≥ ${needs.card} GB`;
  return el("span", { class: "needs" },
    needs.weights ? el("span", {}, `needs ${card} and ≥ ${needs.disk} GB of disk`,
      el("span", { class: "muted" }, ` (${needs.weights} GB of weights)`)) : null,
    needs.unknown.length ? el("span", { class: "muted" },
      `${needs.weights ? " · " : ""}size not measured for ${needs.unknown.join(", ")}, so not counted`) : null);
}

const profileName = (model) => model.replace(/[^A-Za-z0-9 ._-]+/g, "-").slice(0, 40);

// The profiles in the file, or — before a pool has any — the placement in force written as
// profiles, so that saving without touching them changes only the words, not what is rented.
function profilesFromStatus(status, rented) {
  const e = status.engine || {};
  const pool = e.pool_variants || [];
  const about = (model, tag) => pool.find((v) => v.model === model && v.tag === tag) || {};
  const row = (model, tag) => ({ model, build: tag || "", size_gb: about(model, tag).size_gb ?? null, precision: about(model, tag).precision ?? null });
  if ((e.profiles || []).length) {
    return {
      migrated: false,
      profiles: e.profiles.map((p) => ({
        name: p.name, rented: p.rented, several: p.models.length > 1, cards: p.cards_per_copy || 1,
        models: p.models.map((m) => ({ ...row(m.model, m.build), size_gb: m.size_gb ?? about(m.model, m.build).size_gb ?? null })),
      })),
    };
  }
  // The first variant rented machines can actually run: a build for another platform is not one.
  const caps = new Set(e.rented_capabilities || []);
  const first = (model) => (pool.find((v) => v.model === model && v.engine === rented
    && (v.requires || []).every((c) => caps.has(c))) || {}).tag;
  const set = status.model_set || [];
  if (e.models_per_host === "declared") {
    const renting = e.rented_models || set;
    return { migrated: true, profiles: renting.map((m) => ({ name: profileName(m), rented: true, several: false, models: [row(m, first(m))] })) };
  }
  return { migrated: true, profiles: [{ name: "every model", rented: true, several: set.length > 1, models: set.map((m) => row(m, first(m))) }] };
}

// --- finding a model and its variant ---

const finder = {
  query: "", loading: false, error: null, results: null,
  filters: { minParams: "", maxParams: "", maxSize: "", fitsCard: "", precision: "", kind: "", hideGated: true },
  // Names typed for models new to the pool, by the hub model they come from.
  names: {},
  // Exact sizes being read, by repository, so each is asked for once.
  sizes: {},
  reset() { this.error = null; },
};

const paramsOf = (label) => {
  const m = /(\d+(?:\.\d+)?)\s*([mb])$/i.exec(String(label || ""));
  return m ? Number(m[1]) / (m[2].toLowerCase() === "m" ? 1000 : 1) : null;
};

// Whether a variant row passes the filters. What is not known about a row does not exclude it,
// except where the filter is about exactly that: an unmeasured size passes a size filter only
// once it is measured.
function passes(row) {
  const f = finder.filters;
  const num = (v) => (v === "" || v === null || v === undefined ? null : Number(v));
  const [lo, hi, max, card] = [num(f.minParams), num(f.maxParams), num(f.maxSize), num(f.fitsCard)];
  if (lo !== null && row.params != null && row.params < lo) return false;
  if (hi !== null && row.params != null && row.params > hi) return false;
  if (max !== null && row.size != null && row.size > max) return false;
  if (card !== null && row.size != null && needsFor([{ model: "x", size_gb: row.size }]).card > card) return false;
  if (f.precision && row.precision !== f.precision) return false;
  if (f.kind && row.kinds && row.kinds.length && !row.kinds.includes(f.kind)) return false;
  if (f.hideGated && row.gated) return false;
  return true;
}

async function runSearch(draw) {
  const v = finder;
  if (!v.query.trim()) { v.results = null; draw(); return; }
  v.loading = true; v.error = null; draw();
  try {
    v.results = await api.searchModels(v.query.trim());
  } catch (error) {
    v.error = error.message;
  } finally {
    v.loading = false;
    draw();
  }
}

// The weights of a hub variant, read once, then the results redrawn with it — only the results,
// so a search being typed keeps its focus.
function measure(repo, redraw) {
  if (repo in finder.sizes) return finder.sizes[repo];
  finder.sizes[repo] = undefined;
  api.modelSize(repo).then((a) => { finder.sizes[repo] = a.size_gb; }).catch(() => { finder.sizes[repo] = null; }).finally(redraw);
  return undefined;
}

function finderView(status, d, profile, draw) {
  const v = finder;
  const hubEngine = !!(((status.engine.offers || {})[d.rented] || {}).builds_on_hub);
  const f = v.filters;
  const results = el("div", {});
  const redraw = () => results.replaceChildren(...finderResults(status, d, profile, draw, hubEngine, redraw));
  const number = (key, placeholder) => el("input", { type: "number", step: "any", min: "0", style: "width:5rem",
    placeholder, value: f[key], oninput: (ev) => { f[key] = ev.target.value; redraw(); } });
  const select = (key, options) => el("select", { onchange: (ev) => { f[key] = ev.target.value; redraw(); } },
    ...options.map(([value, label]) => el("option", { value, ...(f[key] === value ? { selected: true } : {}) }, label)));
  const search = el("input", { type: "search", class: "finder-search", value: v.query,
    placeholder: hubEngine ? "search every model: qwen3 30b, gemma 4, embed…" : "search Ollama's library: qwen3, gemma4, embed…",
    oninput: (ev) => { v.query = ev.target.value; redraw(); },
    onkeydown: (ev) => { if (ev.key === "Enter") runSearch(draw); } });
  const view = el("div", { class: "finder" },
    el("div", { class: "row", style: "margin-top:0" },
      el("strong", {}, !profile.several && profile.models.length ? "Replace the model" : "Add a model"),
      el("span", { class: "muted" }, "— each row is a model and one variant of it; one click adds both"),
      el("span", { style: "margin-left:auto" }),
      el("button", { class: "small", onclick: () => { d.finding = null; draw(); } }, "Close")),
    el("div", { class: "row" }, search,
      el("button", { class: "primary small", onclick: () => runSearch(draw) }, v.loading ? "Searching…" : "Search")),
    el("div", { class: "row filters" },
      el("span", { class: "muted" }, "parameters"), number("minParams", "min B"), el("span", {}, "–"), number("maxParams", "max B"),
      el("span", { class: "muted" }, "size ≤"), number("maxSize", "GB"),
      el("span", { class: "muted" }, "fits a card of"), number("fitsCard", "GB"),
      hubEngine ? select("precision", [["", "any precision"], ...["BF16", "FP16", "FP8", "NVFP4", "MXFP4", "INT4", "INT8"].map((p) => [p, p])]) : null,
      select("kind", [["", "any kind"], ["chat", "chat"], ["vision", "vision"], ["embedding", "embedding"], ["tools", "tools"], ["thinking", "thinking"]]),
      hubEngine ? el("label", { class: "pick" }, el("input", { type: "checkbox", ...(f.hideGated ? { checked: true } : {}),
        onchange: (ev) => { f.hideGated = ev.target.checked; redraw(); } }), el("span", {}, "hide gated")) : null),
    v.error ? el("p", { class: "error" }, v.error) : null,
    results);
  redraw();
  return view;
}

// One variant row: what it is, and the button that puts it — with its model — in the profile.
function variantRow(cells, onAdd, label) {
  return el("tr", {}, ...cells, el("td", {}, el("button", { class: "small", onclick: onAdd }, label)));
}

function finderResults(status, d, profile, draw, hubEngine, redraw) {
  const v = finder;
  const words = v.query.trim().toLowerCase().split(/\s+/).filter(Boolean);
  const matches = (...texts) => words.every((w) => texts.some((t) => String(t || "").toLowerCase().includes(w)));
  const action = !profile.several && profile.models.length ? "Use" : "Add";
  const held = new Set(profile.models.map((m) => `${m.model}|${m.build}`));
  const out = [];

  // What the pool already has — no search needed.
  const pool = variantsFor(status, d)
    .filter((x) => !held.has(`${x.model}|${x.tag}`) && matches(x.model, x.tag))
    .map((x) => ({ ...x, params: paramsOf((x.model.split(":")[1] || "").replace(/^e/, "")), size: x.size_gb }))
    .filter(passes);
  out.push(el("h4", {}, `In the pool — ${pool.length}`));
  out.push(pool.length ? el("table", { class: "builds" },
    el("thead", {}, el("tr", {}, ...["Model", "Variant", "Precision", "Size", ""].map((h) => el("th", {}, h)))),
    el("tbody", {}, ...pool.map((x) => variantRow([
      el("td", { class: "mono" }, x.model), el("td", { class: "mono" }, x.tag),
      el("td", {}, x.precision || "—"), el("td", { class: "num" }, x.size_gb ? `${x.size_gb} GB` : "—"),
    ], () => { hold(d, profile, x.model, x.tag, x.size_gb, x.precision, false); draw(); }, action))))
    : el("p", { class: "muted" }, words.length ? "Nothing in the pool matches." : `The pool has no other ${d.rented} variant.`));

  if (!v.results) {
    out.push(el("p", { class: "muted" }, `Press Search to look ${hubEngine ? "on the model hub" : "in Ollama's library"} for anything else.`));
    return out;
  }
  if (hubEngine) out.push(...hubResults(status, d, profile, draw, action, held, redraw));
  else out.push(...libraryResults(status, d, profile, draw, action, held));
  return out;
}

// At most this many models from a search, and variants of each, are shown — and measured.
const SHOWN_GROUPS = 8, SHOWN_VARIANTS = 6;

function hubResults(status, d, profile, draw, action, held, redraw) {
  const v = finder;
  const pool = status.engine.pool_variants || [];
  const out = [el("h4", {}, "On the model hub")];
  if (v.results.hub_problem) out.push(el("p", { class: "error" }, v.results.hub_problem));
  let shown = 0;
  for (const group of v.results.hub || []) {
    // A group the pool already serves under a name keeps that name; a new one is named here.
    const known = pool.find((p) => group.variants.some((x) => x.repo === p.tag));
    const suggested = group.model.split("/").pop().toLowerCase();
    const name = known ? known.model : (v.names[group.model] ?? suggested);
    const rows = [];
    for (const x of group.variants) {
      if (held.has(`${name}|${x.repo}`)) continue;
      if (rows.length >= SHOWN_VARIANTS) break;
      const size = x.size_gb ?? (x.why_not ? null : measure(x.repo, redraw));
      const row = { params: group.params_b, size, precision: x.precision, kinds: [group.task].filter(Boolean), gated: x.gated };
      if (!passes(row)) continue;
      rows.push(variantRow([
        el("td", { class: "mono" }, x.repo, x.relation === "original" ? el("span", {}, " ", pill("original", "ok")) : null),
        el("td", {}, x.precision || "?"),
        el("td", { class: "num" }, size ? `${size} GB` : size === null ? "—" : el("span", { class: "muted" }, "measuring…")),
        el("td", { class: "muted" }, x.full_speed_on ? `full speed on ${x.full_speed_on}` : (x.runs_on ? `runs on ${x.runs_on}` : "—")),
        el("td", { class: x.why_not ? "error" : "muted" }, x.why_not || (x.options || []).join(", ")),
      ], x.why_not ? () => {} : () => {
        hold(d, profile, name, x.repo, size || null, x.precision, !pool.some((p) => p.tag === x.repo), x.repo);
        draw();
      }, x.why_not ? "—" : action));
    }
    if (!rows.length) continue;
    shown += 1;
    if (shown > SHOWN_GROUPS) break;
    out.push(el("div", { class: "hub-group" },
      el("div", { class: "row", style: "margin-top:0" },
        el("strong", { class: "mono" }, group.model),
        group.params_b ? el("span", {}, `${group.params_b} B`) : null,
        group.task ? pill(group.task) : null,
        group.made_from ? el("span", { class: "muted" }, group.made_from) : null,
        el("span", { style: "margin-left:auto" }),
        known ? el("span", { class: "muted" }, "in the pool as ", el("span", { class: "mono" }, known.model))
          : el("span", {}, el("span", { class: "muted" }, "add as "),
            el("input", { type: "text", class: "mono", style: "width:14rem", value: name, "aria-label": "name in the pool",
              oninput: (ev) => { v.names[group.model] = ev.target.value; } }))),
      el("table", { class: "builds" },
        el("thead", {}, el("tr", {}, ...["Variant", "Precision", "Size", "Cards", "", ""].map((h) => el("th", {}, h)))),
        el("tbody", {}, ...rows))));
  }
  if (!shown) out.push(el("p", { class: "muted" }, `Nothing on the hub matches "${v.query}" with these filters.`));
  return out;
}

function libraryResults(status, d, profile, draw, action, held) {
  const v = finder;
  const pool = status.engine.pool_variants || [];
  const out = [el("h4", {}, "In Ollama's library",
    v.results.library_refreshing ? el("span", { class: "muted" }, " · being read now; search again in a minute")
      : !v.results.library_read ? el("span", {}, " · never read on this pool ",
        el("button", { class: "small", onclick: async (ev) => {
          ev.target.disabled = true;
          try { await api.refreshDirectory(); v.results.library_refreshing = true; } catch (error) { v.error = error.message; }
          draw();
        } }, "Read it now")) : null)];
  let shown = 0;
  for (const model of v.results.library || []) {
    const rows = [];
    for (const t of model.tags || []) {
      if (t.runtime === "cloud" || t.runtime === "mlx" || held.has(`${t.name}|${t.name}`)) continue;  // rented hosts run neither
      const row = { params: paramsOf((t.name.split(":")[1] || "").replace(/^e/, "")), size: t.size_gb ?? null, kinds: model.capabilities || [] };
      if (!passes(row)) continue;
      rows.push(variantRow([
        el("td", { class: "mono" }, t.name, t.in_pool ? el("span", {}, " ", pill("in the pool", "ok")) : null),
        el("td", { class: "num" }, t.size_gb ? `${t.size_gb} GB` : "—"),
        el("td", {}, t.context || "—"),
      ], () => {
        hold(d, profile, t.name, t.name, t.size_gb ?? null, null, !pool.some((p) => p.tag === t.name));
        draw();
      }, action));
    }
    if (!rows.length) continue;
    shown += 1;
    out.push(el("div", { class: "hub-group" },
      el("div", { class: "row", style: "margin-top:0" }, el("strong", { class: "mono" }, model.name),
        ...(model.capabilities || []).map((c) => pill(c)), el("span", { class: "muted" }, model.description || "")),
      el("table", { class: "builds" },
        el("thead", {}, el("tr", {}, ...["Variant", "Size", "Context", ""].map((h) => el("th", {}, h)))),
        el("tbody", {}, ...rows.slice(0, 12)))));
    if (shown >= 20) break;
  }
  if (!shown) out.push(el("p", { class: "muted" }, `Nothing in the library matches "${v.query}" with these filters.`));
  return out;
}

// Put a model and its variant in a profile: one model per machine replaces, several adds. A
// variant new to the pool is remembered to be written into the catalog on save.
function hold(d, profile, model, tag, size, precision, isNew, repo) {
  if (!/^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$/.test(model || "")) {
    finder.error = `"${model}" cannot be a model's name: letters, digits, '.', '_', ':', '-' and '/'`;
    return;
  }
  finder.error = null;
  const entry = { model, build: tag, size_gb: size, precision };
  if (!profile.several) profile.models = [entry];
  else {
    const at = profile.models.findIndex((m) => m.model === model);
    if (at >= 0) profile.models[at] = entry; else profile.models.push(entry);
  }
  if (isNew) {
    const list = d.adds[model] || (d.adds[model] = []);
    if (!list.some((b) => b.tag === tag)) list.push({ tag, engine: d.rented, size_gb: size, precision });
  }
  // A size the search could not tell is read now, and fills in when it arrives.
  if (!size && repo) {
    api.modelSize(repo).then((a) => {
      if (!a.size_gb) return;
      entry.size_gb = a.size_gb;
      for (const b of d.adds[model] || []) if (b.tag === tag) b.size_gb = a.size_gb;
    }).catch(() => {});
  }
}

// --- machine setup: the engine's image, options and start ---

function machineSetup(status, d, draw) {
  const vllm = d.rented === "vllm";
  const offer = (status.engine.offers || {})[d.rented] || { builds_on_hub: false, options: {} };
  const rows = [];
  if (vllm) {
    const list = el("div", {}, ...(d.images || []).map((img, i) => el("div", { class: "row" },
      el("input", { type: "text", style: "width:18rem", value: img.image,
        oninput: (ev) => { d.images[i].image = ev.target.value; } }),
      el("span", { class: "muted" }, "driver ≥"),
      el("input", { type: "text", style: "width:5rem", value: img.min_driver,
        oninput: (ev) => { d.images[i].min_driver = ev.target.value; } }),
      el("button", { class: "small", onclick: () => { d.images.splice(i, 1); draw(); } }, "Remove"))),
      el("button", { class: "small", onclick: () => { d.images.push({ image: "", min_driver: "" }); draw(); } }, "Add an image"));
    rows.push(stackedRow("images", "images, newest first", { control: list },
      "a machine gets the first its driver can run; one that can run none is never bid on"));
  } else {
    rows.push(stackedRow("image", "image", { control: el("input", { type: "text", style: "width:18rem", value: d.image,
      oninput: (ev) => { d.image = ev.target.value; } }) }, "pinned, never a floating tag"));
  }
  const optionNames = Object.keys(offer.options || {});
  if (optionNames.length) {
    const custom = (d.engine_start || "").trim() !== "";
    rows.push(stackedRow("engine_options", "engine options",
      { control: el("div", {}, ...optionNames.map((key) => optionBox(key, offer.options[key], d.engine_options, custom, draw))) },
      custom ? "these belong to the engine's own start; clear the start command to use them"
        : "each model gets an option only if its family has one — the rest start without it, and the machine says so"));
  }
  rows.push(stackedRow("engine_start", "start command", {
    control: el("input", { type: "text", style: "width:26rem", value: d.engine_start,
      placeholder: vllm ? "blank: the agent's own vllm-start (recommended)" : "how the image's engine is started, if it does not start itself",
      oninput: (ev) => { d.engine_start = ev.target.value; } }),
  }, vllm ? "blank lets the pool start vLLM on what its agent downloaded" : "runs after the dead-man timer is armed"));
  return el("details", { class: "panel machine-setup", ...(engineEdit.setupOpen ? { open: true } : {}),
    ontoggle: (ev) => { engineEdit.setupOpen = ev.target.open; } },
    el("summary", {}, el("strong", {}, "Machine setup"),
      el("span", { class: "muted" }, ` — ${d.rented}: ${vllm ? `${(d.images || []).length} image(s)` : d.image || "no image"}, ${d.engine_options.length ? d.engine_options.join(", ") : "no options"}, ${d.engine_start ? "own start command" : "the engine's own start"}`)),
    el("table", {}, el("tbody", {}, ...rows)),
    enginePanel(status.engine));
}

// --- the engine's named options (D100) ---

const ago = (ts) => {
  if (!ts) return "never";
  const minutes = Math.max(0, (Date.now() / 1000 - ts) / 60);
  if (minutes < 1) return "just now";
  if (minutes < 90) return `${Math.round(minutes)} min ago`;
  if (minutes < 48 * 60) return `${Math.round(minutes / 60)} h ago`;
  return `${Math.round(minutes / 1440)} days ago`;
};

function optionBox(key, option, chosen, disabled, redraw) {
  const box = el("input", { type: "checkbox", ...(chosen.includes(key) ? { checked: true } : {}),
    ...(disabled ? { disabled: true } : {}),
    onchange: (ev) => {
      const at = chosen.indexOf(key);
      if (ev.target.checked && at < 0) chosen.push(key);
      if (!ev.target.checked && at >= 0) chosen.splice(at, 1);
      redraw();
    } });
  return el("label", { class: "pick option" }, box, el("span", {}, option.label),
    el("span", { class: "muted" }, ` — for ${option.families.join(", ")}`));
}

async function saveEngine(event, note) {
  const d = engineEdit.draft;
  // Said here, before the file's own rules would say it less plainly.
  const names = d.profiles.map((p) => (p.name || "").trim());
  const problem = names.some((n) => !n) ? "every profile needs a name"
    : new Set(names).size !== names.length ? "two profiles have the same name"
    : d.profiles.some((p) => !p.models.length) ? `profile "${d.profiles.find((p) => !p.models.length).name}" holds no model`
    : d.profiles.some((p) => p.models.some((m) => !m.build)) ? "choose a build for every model in every profile"
    : !d.profiles.some((p) => p.rented) ? "tick at least one profile to rent as" : null;
  if (problem) {
    engineEdit.message = ` not saved: ${problem}`;
    note.textContent = engineEdit.message;
    return;
  }
  const used = new Set(d.profiles.flatMap((p) => p.models.map((m) => `${m.model}|${m.build}`)));
  const body = {
    rented_engine: d.rented,
    engine_start: d.engine_start,
    engine_options: (d.engine_start || "").trim() ? [] : d.engine_options,
    profiles: Object.fromEntries(d.profiles.map((p) => [p.name.trim(), Object.fromEntries(p.models.map((m) => [m.model, m.build]))])),
    rent: d.profiles.filter((p) => p.rented).map((p) => p.name.trim()),
    // Only where the engine can split; another engine's profiles are sent unsplit (D114).
    split: Object.fromEntries(d.profiles.map((p) => [p.name.trim(),
      (((d.offers || {})[d.rented] || {}).splits_across_cards ? (p.cards || 1) : 1)])),
    // Only builds a profile still uses: one found and then dropped is not written to the file.
    add: Object.entries(d.adds).map(([name, builds]) => ({ name, builds: builds.filter((b) => used.has(`${name}|${b.tag}`)) }))
      .filter((item) => item.builds.length),
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

// Taking a configured host out of the pool (a laptop wanted back, a box being moved). Out of
// service is the reversible one: the host stays in the file and returns with one click. Remove
// takes it out of the file, and is typed to confirm. Both go through the file, the plan and its
// rules, so taking out the only host serving a model is refused with the reason.
function hostControls(host) {
  const out = host.state === "disabled";
  const service = out
    ? el("button", { class: "small", onclick: (e) => run(e.target, () => api.setHostService(host.host_id, false)) },
        "Return to service")
    : el("button", { class: "small", onclick: async (e) => {
        const ok = await confirmAction({
          title: `Take ${host.host_id} out of service?`,
          body: "It stops receiving requests; any already on it finish. It stays in the configuration and returns with one click.",
        });
        if (ok) run(e.target, () => api.setHostService(host.host_id, true));
      } }, "Take out of service");
  const remove = el("button", { class: "small danger", onclick: (e) => run(e.target, async () => {
    try {
      await api.removeHost(host.host_id);  // asks first: the answer is the plan, and what to type
    } catch (error) {
      const retype = (error.changes || []).find((c) => c.requires_retype);
      if (!retype) throw error;  // refused by the pool's rules; the message says why
      const ok = await confirmAction({
        title: `Remove ${host.host_id} from the pool?`,
        body: el("div", {},
          ...(error.changes || []).map((c) => el("p", {}, c.detail)),
          el("p", { class: "muted" }, "Its entry leaves the configuration file. The file keeps its history, so this can be rolled back on the Configuration screen.")),
        retype: retype.value,
      });
      if (ok) await api.removeHost(host.host_id, retype.value);
    }
  }) }, "Remove…");
  return el("div", { class: "row" }, service, remove);
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
  ["interruptible", "interruptible — a bid or a spot price; can be taken away"],
  ["on_demand", "on demand — fixed price; nothing can take it away"],
  ["cheaper", "cheaper — search both listings, take whichever costs less"],
];

// What the next machine is bought as, what that needs, and the minimums the search used because
// of it (D111): an operator's own minimum stands where it is higher, and is raised where not.
function nextHostNote(market, { onlyWhenRaised = false } = {}) {
  const next = market.next_host;
  if (!next || !next.needs) return null;
  const { searched, typed, needs } = next;
  const raised = searched.min_gpu_memory_gb > typed.min_gpu_memory_gb || searched.min_disk_gb > typed.min_disk_gb
    || (searched.gpus_multiple_of || 1) > (typed.gpus_multiple_of || 1);
  if (onlyWhenRaised && !raised) return null;
  const what = next.profile ? `bought as ${next.profile}, holding ${next.models.join(", ")}` : `holding ${next.models.join(", ")}`;
  const cards = (needs.cards_per_copy || 1) > 1
    ? `cards in whole groups of ${needs.cards_per_copy}, each of ≥ ${needs.card_memory_gb} GB,` : `a card of ≥ ${needs.card_memory_gb} GB`;
  return el("div", { class: "muted", style: "margin-top:6px" },
    `The next host is ${what}: it needs ${cards} and ${needs.disk_gb} GB of disk`,
    needs.unknown.length ? ` (${needs.unknown.join(", ")} not measured, not counted)` : "",
    raised ? ` — so the search asks for ≥ ${searched.min_gpu_memory_gb} GB of card and ${searched.min_disk_gb} GB of disk, above the ${typed.min_gpu_memory_gb} and ${typed.min_disk_gb} set here.`
      : " — within the minimums set here.");
}

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
      nextHostNote(market, { onlyWhenRaised: true }),
      el("div", { class: "row" },
        // Deliberately not through `run`: that refreshes the screen, which would rebuild this
        // form from the saved values and throw away what was just typed into it.
        el("button", { class: "primary", onclick: async (e) => {
          const button = e.target, label = button.textContent;
          button.disabled = true; button.textContent = "asking…";
          try {
            const fresh = await api.market(1, searchValues());
            marketView.box.replaceChildren(marketPanel(fresh));
            if (fresh.search_quota) marketView.quota.replaceChildren(searchQuotaText(fresh.search_quota));
            marketView.at = Date.now();
            marketView.last = fresh;
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

// The last search this page made, kept so moving between tabs shows it again without asking
// the provider. There is no timer: a search spends the provider's daily quota (D120).
const marketView = { box: null, stamp: null, button: null, quota: null, busy: false, at: 0, last: null,
  // What the next search asks: every provider (null) or the ones chosen, and which kinds.
  connections: null, kinds: "both" };

// The chips that narrow a search (providers.md §7.2): a provider each, and the two kinds. A
// search still happens only when a button is pressed (D120).
function marketChips(shown) {
  const lines = shown.by_connection || {};
  const choosable = Object.entries(lines).filter(([, line]) => line.asked || line.why_not === "not chosen for this search"
    || line.why_not === "no search yet");
  const chip = (label, on, toggle, title) => el("button", { type: "button", class: `chip${on ? " on" : ""}`,
    "aria-pressed": on ? "true" : "false", title, onclick: (e) => { toggle(); e.target.closest(".chips").replaceWith(marketChips(marketView.last || shown)); } }, label);
  const chosen = (name) => !marketView.connections || marketView.connections.has(name);
  const providers = choosable.length > 1 ? choosable.map(([name]) => chip(name, chosen(name), () => {
    const next = new Set(marketView.connections || choosable.map(([n]) => n));
    if (next.has(name)) { if (next.size > 1) next.delete(name); } else next.add(name);
    marketView.connections = next.size === choosable.length ? null : next;
  }, `search ${name}`)) : [];
  const kinds = { both: ["interruptible", "on_demand"], interruptible: ["interruptible"], on_demand: ["on_demand"] };
  const has = (k) => (kinds[marketView.kinds] || kinds.both).includes(k);
  const kindChip = (k, label) => chip(label, has(k), () => {
    const now = new Set(kinds[marketView.kinds] || kinds.both);
    if (now.has(k)) { if (now.size > 1) now.delete(k); } else now.add(k);
    marketView.kinds = now.size === 2 ? "both" : [...now][0];
  }, label);
  return el("div", { class: "row chips", role: "group", "aria-label": "What the next search asks" },
    el("span", { class: "muted" }, "Search"), ...providers,
    providers.length ? el("span", { class: "muted" }, "·") : null,
    kindChip("interruptible", "interruptible"), kindChip("on_demand", "on demand"));
}

// A line per provider: asked or why not, what it returned, what passed, its quota.
function marketLines(market) {
  const lines = Object.entries(market.by_connection || {});
  if (lines.length < 2) return null;
  return el("table", { class: "market-lines" },
    el("thead", {}, el("tr", {}, ...["Provider", "Seen", "Pass", "Search quota", ""].map((h) => el("th", {}, h)))),
    el("tbody", {}, lines.map(([name, line]) => el("tr", {},
      el("td", {}, name),
      el("td", { class: "num" }, line.asked ? line.seen : "—"),
      el("td", { class: "num" }, line.asked ? line.passed : "—"),
      el("td", { class: "muted small-text" }, line.search_quota ? searchQuotaText(line.search_quota) : "none"),
      el("td", { class: line.error ? "error" : "muted" }, line.error || line.why_not || "")))));
}

function marketSection(settings) {
  const shown = marketView.last || settings;
  marketView.box = el("div", {}, marketPanel(shown));
  marketView.stamp = el("span", { class: "muted" });
  marketView.button = el("button", { class: "small", onclick: () => refreshMarket() }, "Search the market");
  marketView.quota = el("div", {}, searchQuotaText(shown.search_quota));
  if (!marketView.last) marketView.at = 0;
  stampMarket();
  return [
    el("div", { class: "row" },
      el("h2", {}, "Live market — the real offer pipeline, read-only"), marketView.button, marketView.stamp),
    marketView.quota,
    marketChips(shown),
    marketView.box,
  ];
}

const marketOnScreen = () =>
  state.screen === "rented" && marketView.box !== null && document.body.contains(marketView.box);

function stampMarket() {
  if (!marketView.stamp) return;
  marketView.stamp.textContent = marketView.busy
    ? " asking the provider…"
    : marketView.at ? ` searched at ${new Date(marketView.at).toLocaleTimeString()}` : " not searched yet";
}

async function refreshMarket() {
  if (marketView.busy || !marketOnScreen()) return;
  marketView.busy = true;
  marketView.button.disabled = true;
  stampMarket();
  try {
    const fresh = await api.market(1).catch((error) => ({ error: error.message }));
    marketView.last = fresh.error ? marketView.last : fresh;
    if (marketOnScreen()) {  // the operator may have moved on while it was asked
      marketView.box.replaceChildren(marketPanel(fresh));
      if (fresh.search_quota) marketView.quota.replaceChildren(searchQuotaText(fresh.search_quota));
    }
  } finally {
    marketView.at = Date.now();
    marketView.busy = false;
    marketView.button.disabled = false;
    stampMarket();
  }
}

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
  if (market.searched === false) {
    return el("div", { class: "panel" },
      el("p", {}, "The market has not been searched from this page."),
      el("p", { class: "muted" },
        "The provider counts every offer a search returns against a daily quota, and the pool needs that quota to rent "
        + "and to replace hosts — so this page searches only when you press Search the market or Try these."));
  }
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
      el("p", { class: "muted" }, "Nothing can be rented until a provider answers. A provider may limit searches by a daily quota of offers returned — once it is spent, every search is refused until it resets."));
  }
  const unasked = Object.entries(market.provider_errors || {});
  const unaskedNote = unasked.length ? el("p", { class: "warn-text" },
    `${unasked.map(([name, why]) => `${name} could not be asked (${why})`).join("; ")} — the offers below are the other providers' only.`) : null;
  // Rejections with a column per provider, where more than one was asked.
  const askedNames = Object.entries(market.by_connection || {}).filter(([, l]) => l.asked).map(([n]) => n);
  const perProvider = askedNames.length > 1;
  return el("div", {}, unaskedNote, avoidedNote, marketLines(market), el("div", { class: "grid" },
    el("div", { class: "panel" },
      el("div", { class: "stat" }, `${market.passed} pass · ${market.rejected} rejected`),
      el("div", { class: "muted" }, `${market.seen} offers seen through your policy`),
      // Asked of the provider (D123): an offer failing these is never returned, so it is not
      // among the rejections below — and is not counted against the daily search quota.
      (market.filtered_by_provider || []).length ? el("div", { class: "muted" },
        `The provider was asked for only: ${market.filtered_by_provider.join(" · ")}`) : null,
      // What the disk has to hold, from the builds' own sizes (D108) — beside the disk the
      // host is rented with, so the two can be compared at a glance.
      market.policy ? el("div", { class: "muted" },
        `rented with ${market.policy.disk_gb} GB of disk · the models it would be bought for take ` +
        `${market.model_set_gb} GB` +
        ((market.model_sizes_unknown || []).length
          ? ` (not measured yet: ${market.model_sizes_unknown.join(", ")})` : "")) : null,
      nextHostNote(market),
      el("h2", {}, "Rejected, by reason"),
      el("table", {},
        perProvider ? el("thead", {}, el("tr", {}, el("th", {}, "Reason"),
          ...askedNames.map((n) => el("th", { class: "num" }, n)))) : null,
        el("tbody", {}, Object.entries(market.rejected_by_reason || {}).map(([reason, count]) => perProvider
          ? el("tr", {}, el("td", { class: "muted" }, reason),
            ...askedNames.map((n) => el("td", { class: "num" }, (market.by_connection[n].rejected_by_reason || {})[reason] || 0)))
          : el("tr", {}, el("td", { class: "num" }, count), el("td", { class: "muted" }, reason)))))),
    el("div", { class: "panel wide" }, el("h2", {}, "Best offers"),
      el("table", {},
        el("thead", {}, el("tr", {},
          el("th", {}, "Hardware"), el("th", {}, "Kind"), el("th", { class: "num" }, "Floor"), el("th", { class: "num" }, "Would pay"),
          el("th", { class: "num" }, "On-demand"), el("th", { class: "num" }, "$/GB"),
          el("th", { class: "num", title: "Workers a host rented from this offer would run" }, "Workers"),
          // What the list is ordered by, and what the pool rents by (D131); the score only filters.
          el("th", { class: "num", title: "Expected cost per worker-hour: the price, interruptions, the download and this machine's record here — the pool rents the lowest" }, "Per worker-hour"),
          el("th", {}, ""))),
        el("tbody", {}, (market.best || []).map((offer, index) => el("tr", {},
          el("td", {}, offer.would_rent ?? index === 0 ? el("strong", {}, offer.hardware) : offer.hardware,
            offer.would_rent ? el("span", { title: "the offer the pool's own rule picks, across every provider asked" }, " ", pill("would rent", "ok")) : null,
            el("div", { class: "muted mono" }, `${offer.machine} · ${offer.gpu_memory_gb}GB · `,
              // A value the provider does not report is marked as assumed, never shown as measured (§6).
              (offer.assumed || []).includes("download_mbps")
                ? el("span", { title: "assumed: this provider does not report download speed" }, `≈${offer.download_mbps}Mbps`)
                : `${offer.download_mbps}Mbps`)),
          // A bid or a spot price can be taken away; a fixed price cannot. Same machine, different deal.
          el("td", {}, pricedPill(offer), offer.connection ? el("div", { class: "muted" }, offer.connection) : null),
          el("td", { class: "num" }, pricedOf(offer) === "bid" ? rate(offer.floor) : "—"),
          el("td", { class: "num" }, el("strong", {}, rate(offer.would_bid))),
          el("td", { class: "num" }, rate(offer.on_demand)),
          el("td", { class: "num" }, `$${Number(offer.download_per_gb).toFixed(4)}`),
          // A profile's number is shown plainly; the default is marked, so it reads as unmeasured.
          el("td", { class: "num", title: offer.workers_from || "" },
            offer.workers ?? "—",
            offer.workers_from && offer.workers_from.startsWith("capacity profile") ? "" : el("span", { class: "muted" }, " default")),
          el("td", { class: "num", title: `score ${Math.round(offer.score)}` },
            offer.per_worker_hour == null ? "—" : `$${Number(offer.per_worker_hour).toFixed(4)}`),
          el("td", {}, offer.offer_id
            ? el("button", { class: "small", onclick: (e) => rentThis(e, offer) }, "Rent")
            : null))))))));
};

// Renting one particular offer, the way it is listed: this machine, as a bid or at its fixed
// price. It is still the pool's policy deciding what may be rented — the list only shows what
// passes — and if the offer has gone by the time it is asked for, nothing is rented instead.
// How a host or offer is paid for (D132): a bid the pool sets, the provider's spot price, or a
// fixed on-demand price. Both bid and spot can be taken away; on demand cannot.
const PRICED = {
  bid: ["bid", "warn", "a bid — can be outbid at any moment"],
  spot: ["spot", "warn", "spot — the provider's price, which can change; it can be taken back at any time"],
  on_demand: ["on demand", "ok", "on demand — a fixed price; nothing can take it away"],
};
function pricedOf(row) {
  return row.priced || (row.interruptible === false || row.kind === "on_demand" ? "on_demand" : "bid");
}
function pricedPill(row) {
  const [label, tone] = PRICED[pricedOf(row)] || PRICED.bid;
  return pill(label, tone);
}

async function rentThis(event, offer) {
  const spend = Number(document.getElementById("prepare-spend")?.value || 1);
  const hours = Number(document.getElementById("prepare-hours")?.value || 1);
  const when = document.getElementById("prepare-when")?.value || "join";
  const priced = pricedOf(offer);
  const ok = await confirmAction({
    title: `Rent ${offer.hardware}${offer.connection ? ` from ${offer.connection}` : ""}?`,
    body: el("div", {},
      el("p", {}, priced === "on_demand"
        ? `On demand at ${rate(offer.would_bid)} — a fixed price. Nobody can take it away; it runs until the lease ends or you release it.`
        : priced === "spot"
          ? `At the spot price, ${rate(offer.would_bid)}, set by the provider. It can change while the host runs, and the host can be taken back at any time.`
          : `A bid of ${rate(offer.would_bid)} (floor ${rate(offer.floor)}). Cheaper, and it can be outbid at any moment.`),
      el("p", {}, `Worst case: ${money(spend)} over ${hours}h, this one host. Machine ${offer.machine} · ${offer.gpu_memory_gb} GB · ${offer.workers} workers.`),
      el("p", { class: "muted" }, "Cap and time limit come from the Prepare a host panel above. If this offer has gone, nothing else is rented in its place.")),
  });
  if (!ok) return;
  run(event.target, () => api.prepare({
    max_spend: spend, max_hours: hours, when_ready: when, offer_id: offer.offer_id, kind: offer.kind,
    connection: offer.connection,
  }));
}

function preparePanel() {
  const spend = el("input", { id: "prepare-spend", type: "number", step: "0.01", value: "1.00", style: "width:6rem" });
  const hours = el("input", { id: "prepare-hours", type: "number", step: "0.5", value: "1", style: "width:5rem" });
  const when = el("select", { id: "prepare-when" }, el("option", { value: "join" }, "join the pool"), el("option", { value: "park" }, "park it"), el("option", { value: "destroy" }, "destroy"));
  const kind = el("select", {},
    el("option", { value: "" }, "as the pool is configured"),
    el("option", { value: "interruptible" }, "interruptible — a bid or a spot price; cheaper, can be taken away"),
    el("option", { value: "on_demand" }, "on demand — fixed price; nothing can take it away"));
  return el("div", { class: "panel" }, el("h2", {}, "Prepare a host"),
    el("p", { class: "muted" }, "Its own small lease: rent, load the model set, verify, then join, park or destroy. Borrows no authority from any other lease."),
    el("label", {}, "Dollar cap ", spend),
    el("label", {}, "Time limit (hours) ", hours),
    el("label", {}, "When ready ", when),
    el("label", {}, "Rent it ", kind),
    el("p", { class: "muted" }, "Or pick one machine: every offer on the ", el("a", { href: "#rented/finding" }, "Finding machines"), " tab has its own Rent button."),
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
    el("p", { class: "muted" }, "What each host serves now: the variant of each model it was given, and whether it is loaded. A host missing what it was given is not routed to."),
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
    el("p", { class: "muted" }, "Models are added, and the variant each rented machine holds is chosen, under ",
      el("a", { href: "#rented/engine" }, "Rented capacity → Profiles"), "."),
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

// --- workloads (D115, docs/spec/workloads.md) ---

// The form, the last plan, a key just shown and which row is open, kept across redraws. The
// table refreshes itself every few seconds; the form and a key being copied are never redrawn
// under the operator.
const workloadDraft = {
  // One row per model, each with its own target (D118).
  name: "", rows: [{ model: "", latency: "30", parallel: "8" }], placement: "auto", hours: "4", budget: "", kind: "roi",
  plan: null, planning: false, error: null, notice: null, shown: null, open: null,
  rowErrors: {}, timer: null, holder: null,
  // The form folds away once any workload is active, so the table stays in view; it opens on
  // request, and stays open while it holds a draft.
  formOpen: false, formHolder: null, hasLive: null,
};
const KIND_LABELS = { roi: "Cheapest by the numbers", on_demand: "On demand only", interruptible: "Interruptible only (bid or spot)" };
const KIND_HINTS = {
  roi: "on demand, a bid or a spot price — whichever is expected to cost less over its hours, interruptions counted",
  on_demand: "nothing can take it away; the listed price",
  interruptible: "cheapest; can be taken away, and a replacement is rented",
};
const WORKLOAD_STATES = { preparing: "warn", serving: "ok", ending: "", ended: "" };
const NAME_RULE = /^[a-z0-9][a-z0-9-]{0,39}$/;
const dollars = (n) => (n === null || n === undefined ? "—" : `$${Number(n).toFixed(2)}`);
const until = (ts) => (ts ? new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "—");

// One set of names everywhere — form, plan, table, CLI, spec (D118).
const PLACEMENT_LABELS = { auto: "Whichever costs less", together: "Together — every host holds every model",
  apart: "Apart — each model on hosts of its own" };
const PLACEMENT_HINTS = {
  auto: "priced both ways when you plan; the cheaper is kept (within 1% they count as equal: together)",
  together: "each host keeps a fixed share of its answers for each model",
  apart: "each model scales on hosts that hold only it",
};

function workloadBody() {
  const d = workloadDraft;
  const body = { name: d.name.trim(), hours: Number(d.hours), kind: d.kind };
  if (d.rows.length === 1) {
    const [row] = d.rows;
    Object.assign(body, { model: row.model, latency_s: Number(row.latency), parallel: Number(row.parallel) });
  } else {
    body.models = d.rows.map((r) => ({ model: r.model, latency_s: Number(r.latency), parallel: Number(r.parallel) }));
    body.placement = d.placement;
  }
  if (d.budget.trim() !== "") body.max_spend = Number(d.budget.replace(/^\$/, ""));
  return body;
}

// A model rented hosts can run: a catalog build for the rented engine, or one for any engine.
function rentable(status, model) {
  const builds = (status.catalog || {})[model];
  if (!builds || !builds.length) return true;
  const engine = (status.engine || {}).rented;
  return builds.some((b) => b.engine == null || b.engine === engine);
}

function workloadForm(status) {
  const d = workloadDraft;
  const onlyForWorkloads = status.workload_models || [];
  const models = [...new Set([...onlyForWorkloads, ...(status.model_set || [])])];
  const firstFree = () => models.find((m) => rentable(status, m) && !d.rows.some((r) => r.model === m))
    || models.find((m) => !d.rows.some((r) => r.model === m)) || models[0];
  for (const row of d.rows) if (!row.model && models.length) row.model = firstFree();
  const label = (m) => `${m}${onlyForWorkloads.includes(m) ? " (workloads only)" : ""}${rentable(status, m) ? "" : " (no build for rented hosts)"}`;
  let planButton, planHolder, errorHolder, nameCheck, dupNote;
  const rowSelects = [];
  const mostModels = Math.min(status.workload_max_models || 4, models.length);
  const nameOk = () => NAME_RULE.test(d.name.trim()) && !d.name.trim().startsWith("rented-");
  const twice = () => new Set(d.rows.map((r) => r.model)).size !== d.rows.length;
  const sync = (focusPlan) => {
    // A model another row holds cannot be picked again: duplicates are prevented, not reported.
    rowSelects.forEach((select, i) => {
      for (const option of select.options) option.disabled = d.rows.some((r, j) => j !== i && r.model === option.value);
      select.setAttribute("aria-invalid", d.rows.some((r, j) => j !== i && r.model === d.rows[i].model) ? "true" : "false");
    });
    dupNote.textContent = twice() ? "Each model once: pick another for the repeated row." : "";
    planButton.disabled = d.planning || !nameOk() || twice();
    planButton.textContent = d.planning ? "Working it out…" : "Plan";
    planButton.title = !nameOk() ? "Name it first" : twice() ? "Each model once" : "";
    nameCheck.textContent = d.name && !nameOk() ? "lower-case letters, digits and -, not starting with rented-" : "";
    planHolder.replaceChildren(...(d.plan ? [workloadPlanView(d.plan)] : []));
    errorHolder.textContent = d.error || "";
    if (focusPlan && d.plan) planHolder.querySelector("[tabindex]")?.focus();
  };
  const changed = () => { d.plan = null; d.error = null; d.notice = null; sync(); };
  let hintId = 0;
  const wrap = (label, control, hint) => {
    const id = `wl-hint-${++hintId}`;
    if (hint) control.setAttribute("aria-describedby", id);
    return el("label", { class: "field" }, el("span", {}, label), control, hint ? el("small", { class: "muted", id }, hint) : null);
  };
  const input = (key, attrs) => el("input", { ...attrs, value: d[key], oninput: (e) => { d[key] = e.target.value; changed(); } });
  const select = (key, options) => el("select", { onchange: (e) => { d[key] = e.target.value; changed(); } },
    ...options.map(([v, text]) => el("option", { value: v, ...(v === d[key] ? { selected: true } : {}) }, text)));
  // A row's own fields: bound to that row, not to the draft's top level. Their hints are said
  // once, under the rows, and every row's fields point at them.
  const rowHints = "wl-row-hints";
  const rowField = (text, control) => {
    control.setAttribute("aria-describedby", rowHints);
    return el("label", { class: "field" }, el("span", {}, text), control);
  };
  const rowInput = (row, key, attrs) => el("input", { ...attrs, value: row[key], oninput: (e) => { row[key] = e.target.value; changed(); } });
  const rowSelect = (row) => {
    const select = el("select", { onchange: (e) => { row.model = e.target.value; changed(); } },
      ...models.map((m) => el("option", { value: m, ...(m === row.model ? { selected: true } : {}) }, label(m))));
    rowSelects.push(select);
    return select;
  };
  const several = d.rows.length > 1;
  const modelRows = d.rows.map((row, i) => el("div", { class: "model-row", role: "group",
    "aria-label": several ? `Model ${i + 1} of ${d.rows.length}` : "Model" },
    rowField("Model", rowSelect(row)),
    rowField("Answer latency (p95, s)", rowInput(row, "latency", { type: "number", min: "0.1", step: "any" })),
    rowField("Answers at once", rowInput(row, "parallel", { type: "number", min: "1", step: "1" })),
    several ? el("button", { class: "small", "aria-label": `Remove model ${i + 1}`,
      onclick: () => { d.rows.splice(i, 1); d.plan = null; d.focusRow = Math.min(i, d.rows.length - 1); drawWorkloadForm(); } },
    "Remove") : null));
  const full = d.rows.length >= mostModels;
  const addModel = el("button", { class: "small", disabled: full,
    onclick: () => { d.rows.push({ model: firstFree(), latency: "30", parallel: "4" }); d.plan = null; d.focusRow = d.rows.length - 1; drawWorkloadForm(); } },
  "Add model");
  const placementHint = el("small", { class: "muted" }, PLACEMENT_HINTS[d.placement]);
  const placementSelect = select("placement", Object.entries(PLACEMENT_LABELS));
  placementSelect.addEventListener("change", () => { placementHint.textContent = PLACEMENT_HINTS[d.placement]; });
  const kindHint = el("small", { class: "muted" }, KIND_HINTS[d.kind]);
  const kindSelect = select("kind", Object.entries(KIND_LABELS));
  kindSelect.addEventListener("change", () => { kindHint.textContent = KIND_HINTS[d.kind]; });
  nameCheck = el("small", { class: "error", role: "status" });
  dupNote = el("small", { class: "error", role: "status" });
  planButton = el("button", { onclick: () => planWorkload(sync) }, "Plan");
  planHolder = el("div");
  errorHolder = el("span", { class: "error", role: "status" });
  const panel = el("div", { class: "panel" },
    el("h2", {}, "New workload"),
    el("p", { class: "muted" },
      "One or more models, each at its own latency and answers at once, for a time. It gets its own hosts, its own lease and its own key; nothing is rented until you create it."),
    el("div", { class: "form-grid" },
      el("div", { class: "field" }, wrap("Name", input("name", { type: "text", maxlength: "40", placeholder: "e.g. research-run",
        autocomplete: "off", pattern: "[a-z0-9][a-z0-9-]{0,39}" }), "lower-case letters, digits and -"), nameCheck)),
    el("div", { class: "model-rows", role: "group", "aria-label": "Models" }, ...modelRows,
      el("small", { class: "muted", id: rowHints },
        "Latency: the whole answer at p95, as the app receives it. Answers at once: the most answers of this model at the same time, across its hosts. A workloads-only model is never fetched by the shared hosts."),
      el("div", { class: "row" }, addModel,
        full ? el("small", { class: "muted" }, models.length <= d.rows.length ? "every model is already in it"
          : `at most ${mostModels} models in one workload`) : null, dupNote)),
    el("div", { class: "form-grid" },
      several ? el("label", { class: "field" }, el("span", {}, "Placement"), placementSelect, placementHint) : null,
      wrap("Hours", input("hours", { type: "number", min: "0.5", step: "0.5" }), "then its hosts are released"),
      wrap("Budget ($)", input("budget", { type: "number", min: "0.01", step: "0.01", placeholder: "propose one for me" }),
        "its dollar cap; left empty, the plan proposes one"),
      el("label", { class: "field" }, el("span", {}, "Machines"), kindSelect, kindHint)),
    el("div", { class: "row" }, planButton, errorHolder),
    d.notice ? el("p", { class: "warn-text", role: "status" }, d.notice) : null,
    planHolder);
  sync();
  if (d.focusRow != null) {
    const at = d.focusRow;
    d.focusRow = null;
    setTimeout(() => rowSelects[at]?.focus(), 0);
  }
  return panel;
}

async function planWorkload(sync) {
  const d = workloadDraft;
  d.planning = true; d.error = null; sync();
  try {
    d.plan = (await api.planWorkload(workloadBody())).plan;
  } catch (error) {
    d.plan = null; d.error = error.message;
  } finally {
    d.planning = false; sync(true);
  }
}

function workloadPlanView(plan) {
  if (plan.hosts_at_start === undefined) {
    const counts = Object.entries(plan.rejected_by_reason || {}).slice(0, 4).map(([why, n]) => `${n} ${why}`);
    return el("div", { class: "plan refused", tabindex: "-1" }, el("strong", {}, "Cannot start: "), plan.refused,
      counts.length ? el("p", {}, "Machines turned away: ", counts.join(", "), ". See ",
        el("a", { href: "#rented/finding" }, "Rented capacity → Finding machines"), ".") : null);
  }
  const first = plan.first_host;
  const hosts = plan.hosts_at_start;
  const perHour = plan.hourly_total ?? hosts * first.hourly;
  const bid = first.kind !== "on_demand";
  const createButton = el("button", { class: "primary", disabled: Boolean(plan.refused),
    onclick: (e) => createWorkload(e.target) }, `Create — up to ${dollars(plan.max_spend)}`);
  const several = (plan.models || []).length > 1;
  const measured = plan.workers_measured ? pill("measured on this card", "ok")
    : el("span", { class: "muted" }, "— latency not measured on this card yet: sized from its rated capacity");
  const placementNames = { together: "Together — every host holds every model", apart: "Apart — each model on hosts of its own" };
  const hostsWord = (n) => `${n} host${n === 1 ? "" : "s"}`;
  // Why the kept placement was kept, in one line: asked for, cheaper, equal, or the only one the
  // pool's caps allow.
  const priced = plan.placements || {};
  const other = Object.entries(priced).find(([name]) => name !== plan.placement);
  const kept = priced[plan.placement] || {};
  let keptWhy = "as chosen";
  if (other) {
    const [otherName, o] = other;
    if (o.refused) keptWhy = `${otherName} is not possible here`;
    else if (kept.expected <= o.expected) keptWhy = `${dollars(o.expected - kept.expected)} less than ${otherName}`;
    else keptWhy = `within 1% of ${otherName}: equal, so together is kept`;
  }
  const splitLine = (g) => {
    const kind = `${g.first_host.kind !== "on_demand" ? "a bid" : "on demand"}, ${g.first_host.hardware}, ${rate(g.first_host.hourly)}`;
    if (!Object.keys(g.caps || {}).length) {
      return `${g.models[0]} — ${hostsWord(g.hosts_at_start)} × ${g.workers_per_host} answers at once (${g.hosts_at_start * g.workers_per_host} in all) · ${kind}`;
    }
    const each = g.models.map((m) => `${g.caps[m]} ${m}`).join(" and ");
    const all = g.models.map((m) => `${g.caps[m] * g.hosts_at_start} ${m}`).join(", ");
    return `${g.models.join(" + ")} — ${hostsWord(g.hosts_at_start)}, each takes at most ${each} answer${g.workers_per_host === 1 ? "" : "s"} at once (${all} in all) · ${kind}`;
  };
  // Several models (D118): both placements as priced, which was kept and why, and each group's split.
  const placementRows = several ? [
    el("div", { class: "k" }, "placement"),
    el("div", {}, el("strong", {}, placementNames[plan.placement] || plan.placement), ` — kept: ${keptWhy}`,
      ...Object.entries(priced).map(([name, o]) => el("div", { class: "muted" },
        `${name}: ${o.refused ? `not possible — ${o.refused}` : `${hostsWord(o.hosts)}, ${dollars(o.expected)} expected over ${plan.hours} h`}`
        + (name === plan.placement ? " (kept)" : "")))),
    el("div", { class: "k" }, "starts on"),
    el("div", {}, ...plan.groups.map((g) => el("div", {}, splitLine(g))), measured,
      plan.groups.some((g) => Object.keys(g.caps || {}).length)
        ? el("div", { class: "muted" }, "Fixed shares: a request past its model's share on a host waits, even while another model's share there is idle.")
        : null),
  ] : [
    el("div", { class: "k" }, "starts on"),
    el("div", {}, `${hosts} host${hosts === 1 ? "" : "s"}, ${plan.workers_per_host} answers at once each `, measured),
    el("div", { class: "k" }, "first host"),
    el("div", {}, `${bid ? "a bid" : "on demand"}: ${first.hardware}, ${rate(first.hourly)}`,
      bid ? el("span", { class: "muted" }, " — a bid can be taken away; the pool rents a replacement") : null),
  ];
  const borrowing = Object.entries(plan.borrow_by_model || {});
  const listed = (names) => `${names.join(", ")} ${names.length === 1 ? "is" : "are"}`;
  const whileStarting = several && borrowing.some(([, yes]) => yes) && borrowing.some(([, yes]) => !yes)
    ? ` — meanwhile ${listed(borrowing.filter(([, yes]) => yes).map(([m]) => m))} served on shared hosts; ${listed(borrowing.filter(([, yes]) => !yes).map(([m]) => m))} refused until ${plan.placement === "together" ? "the workload's first host is" : "its own hosts are"} ready`
    : plan.borrow_while_starting
      ? ` — meanwhile its requests are served on shared hosts that hold ${several ? "these models" : "this model"}`
      : ` — until then its requests are refused: no shared host serves ${several ? "these models" : "this model"}`;
  return el("div", { class: "plan" + (plan.refused ? " refused" : ""), tabindex: "-1" },
    el("div", { class: "kv" },
      ...placementRows,
      el("div", { class: "k" }, "cost per hour"),
      el("div", {}, !several && Math.abs(hosts * first.hourly - perHour) < 0.005
        ? `${hosts} host${hosts === 1 ? "" : "s"} × ${rate(first.hourly)} = ${rate(perHour)}` : rate(perHour)),
      el("div", { class: "k" }, "budget"),
      el("div", {}, el("strong", {}, dollars(plan.max_spend)),
        plan.budget_derived ? ` — ${rate(perHour)} × ${plan.hours} h, plus 25%` : " — as typed"),
      plan.pool_burn_cap ? el("div", { class: "k" }, "pool burn") : null,
      plan.pool_burn_cap ? el("div", {}, `${rate(plan.pool_burn_after)} of the ${rate(plan.pool_burn_cap)} cap after this (now ${rate(plan.pool_burn_now)})`) : null,
      el("div", { class: "k" }, "serving in about"),
      el("div", {}, `${Math.round(plan.minutes_to_serve)} minutes`, el("span", { class: "muted" }, whileStarting)),
      el("div", { class: "k" }, "at the end"),
      el("div", {}, "When the hours or the budget run out, its hosts are released and its key gets 503 workload_ended.")),
    el("details", {}, el("summary", {}, "How this was worked out"),
      el("ul", {}, ...(plan.reasons || []).map((line) => el("li", {}, line)),
        ...(plan.sizing || []).map((line) => el("li", {}, line)),
        ...(first.reasons || []).map((line) => el("li", {}, line)))),
    plan.refused ? el("p", { class: "error" }, el("strong", {}, "Cannot start: "), plan.refused) : null,
    el("div", { class: "row" }, createButton));
}

async function createWorkload(button) {
  const d = workloadDraft;
  const plan = d.plan;
  const body = workloadBody();
  const budget = plan.max_spend.toFixed(2);
  const ok = await confirmAction({
    title: `Create workload ${body.name}?`,
    body: el("div", {},
      el("p", {}, `This opens a lease that may spend up to $${budget} over ${body.hours} h, and rents ${plan.hosts_at_start} host${plan.hosts_at_start === 1 ? "" : "s"} now.`)),
    retype: plan.budget_derived ? budget : null,
    retypeLabel: `Type ${budget} to accept this budget`,
    okLabel: "Create workload",
  });
  if (!ok) return;
  if (plan.budget_derived) body.confirm_max_spend = Number(budget);
  button.disabled = true;
  try {
    const made = await api.createWorkload(body);
    const targets = body.models || [{ model: body.model, latency_s: body.latency_s, parallel: body.parallel }];
    d.shown = { name: body.name, model: targets.map((t) => t.model).join(", "), connection: made.connection, rotated: false,
                endsAt: made.workload.ends_at, targets };
    Object.assign(d, { name: "", budget: "", plan: null, error: null, notice: null, formOpen: false,
      rows: [{ model: "", latency: "30", parallel: "8" }], placement: "auto" });
  } catch (error) {
    if (plan.budget_derived && /derived one is/.test(error.message)) {
      // The market moved between the plan and now: plan again and say so, rather than dead-end.
      const before = budget;
      try {
        d.plan = (await api.planWorkload(workloadWithout(body))).plan;
        d.notice = `The price moved: the proposed budget is now ${dollars(d.plan.max_spend)} (was $${before}). Review it and create again.`;
      } catch (again) { d.error = again.message; }
    } else {
      d.error = error.message;
    }
  } finally {
    button.disabled = false;
    render();
  }
}

const workloadWithout = (body) => { const copy = { ...body }; delete copy.confirm_max_spend; return copy; };

function copyButton(text, label = "Copy", ariaLabel = null) {
  return el("button", { class: "small", ...(ariaLabel ? { "aria-label": ariaLabel } : {}), onclick: async (e) => {
    try { await navigator.clipboard.writeText(text); e.target.textContent = "Copied"; }
    catch { e.target.textContent = "Select it and copy"; }
    setTimeout(() => { e.target.textContent = label; }, 1500);
  } }, label);
}

// Leaving the page while a key is on it loses the key for good.
const keepKey = (e) => { if (workloadDraft.shown || programDraft.shown) { e.preventDefault(); e.returnValue = ""; } };
window.addEventListener("beforeunload", keepKey);

function keyPanel() {
  const shown = workloadDraft.shown;
  if (!shown) return null;
  const c = shown.connection;
  const targets = shown.targets || [];
  const several = targets.length > 1;
  const handoff = [
    `base_url: ${c.base_url}`, `api_key: ${c.api_key}`, `${several ? "models" : "model"}: ${shown.model}`,
    shown.endsAt ? `valid until: ${until(shown.endsAt)} (while the workload runs)` : null,
    ...targets.map((t) => `${several ? `${t.model}: ` : ""}at most ${t.parallel} answers at once, target ${t.latency_s} s per answer`),
  ].filter(Boolean).join("\n");
  const heading = el("h2", { tabindex: "-1" }, shown.rotated ? `New key for ${shown.name}` : `${shown.name} is created — give its app owner these`);
  const panel = el("div", { class: "panel key-panel", "aria-live": "polite" },
    heading,
    el("p", {}, el("strong", {}, "The key is shown this once."), " Only its hash is kept; if it is lost, mint a new one."),
    el("div", { class: "kv" },
      el("div", { class: "k" }, "base_url"), el("div", { class: "row" }, el("code", {}, c.base_url), copyButton(c.base_url, "Copy", "Copy base URL")),
      el("div", { class: "k" }, "api_key"), el("div", { class: "row" }, el("code", { class: "secret" }, c.api_key), copyButton(c.api_key, "Copy", "Copy API key")),
      el("div", { class: "k" }, several ? "models" : "model"),
      several
        ? el("div", {}, ...targets.map((t) => el("div", { class: "row" }, el("code", {}, t.model),
          copyButton(t.model, "Copy", `Copy model name ${t.model}`),
          el("span", { class: "muted" }, `up to ${t.parallel} at once, p95 ${t.latency_s} s`))))
        : el("div", { class: "row" }, el("code", {}, shown.model), copyButton(shown.model, "Copy", "Copy model name"))),
    el("p", { class: "muted" },
      several ? "One URL and one key for all its models; each request names its model. "
        : "Any OpenAI-compatible client takes these: base URL, API key, and the model name. ",
      c.tls ? "The client must trust the pool's certificate: a public one, or give the app owner the pool's CA file."
        : "The pool listens without TLS here, so the key must not leave this machine."),
    shown.rotated ? el("p", { class: "muted" }, `The old key keeps working until ${until(shown.graceUntil)}.`) : null,
    el("div", { class: "row" },
      copyButton(handoff, "Copy all for the app owner"),
      el("button", { onclick: async () => {
        const ok = await confirmAction({ title: "Put the key away?",
          body: "It will not be shown again. If it is lost, the workload needs a new key.", okLabel: "I have saved it" });
        if (ok) { workloadDraft.shown = null; render(); }
      } }, "I have saved it")));
  // Focused once, when the key first appears — not on every redraw.
  if (!shown.focused) { shown.focused = true; setTimeout(() => heading.focus(), 0); }
  return panel;
}

function latencyCell(w) {
  const answers = w.answers || {};
  const byModel = Object.entries(answers.by_model || {});
  if (byModel.length > 1) {
    // One line per model, against its own target (D118).
    return el("td", { class: "num" }, ...byModel.map(([model, a]) => {
      const over = a.meets_target === false;
      const said = !a.count ? "no answers yet" : a.count < 20 ? `too few (${a.count})` : `${a.p95_s.toFixed(1)} s`;
      // Said in words, not only in colour.
      return el("div", { class: over ? "error" : "" }, `${model}: ${said}`,
        el("span", { class: over ? "" : "muted" }, ` of ${a.latency_s} s target${over ? " — over" : ""}`));
    }));
  }
  if (!answers.count) return el("td", { class: "num" }, "—", el("div", { class: "muted" }, `target ${w.latency_s} s`));
  if (answers.count < 20) {
    return el("td", { class: "num" }, "too few answers yet", el("div", { class: "muted" }, `${answers.count} · target ${w.latency_s} s`));
  }
  const over = answers.meets_target === false;
  return el("td", { class: "num" }, `${answers.p95_s.toFixed(1)} s · ${answers.count} answers`,
    el("div", { class: over ? "error" : "muted" }, over ? `over target (${w.latency_s} s)` : `within ${w.latency_s} s`));
}

function spendCell(w) {
  const cap = w.lease.max_spend || 0;
  const share = cap ? Math.min(1, (w.lease.spent || 0) / cap) : 0;
  return el("td", { class: "num" }, `${dollars(w.lease.spent)} of ${dollars(cap)}`,
    el("div", { class: `burn${share >= 0.95 ? " bad" : share >= 0.8 ? " warn" : ""}`,
      role: "img", "aria-label": `${Math.round(share * 100)}% of its budget spent` },
      el("div", { style: `width:${(share * 100).toFixed(1)}%` })));
}

// Whichever limit comes first: its hours, or its budget at what it costs now.
function timeLeftCell(w) {
  if (!w.lease.open) {
    return el("td", { class: "num" }, w.ended_at ? `ended ${until(w.ended_at)}` : "—");
  }
  const perHour = w.hosts.reduce((sum, h) => sum + (h.hourly || 0), 0);
  const money = perHour > 0 ? Math.max(0, (w.lease.max_spend - w.lease.spent)) / perHour : Infinity;
  const hours = w.lease.hours_left;
  return money < hours
    ? el("td", { class: "num" }, `${money.toFixed(1)} h`, el("div", { class: "warn-text" }, "the budget runs out first"))
    : el("td", { class: "num" }, `${hours.toFixed(1)} h`, el("div", { class: "muted" }, `until ${until(w.ends_at)}`));
}

// While it starts, where each model's requests go now: its own hosts, shared hosts, or refused.
function modelStartLines(w) {
  const ready = new Set(w.hosts.filter((h) => h.state === "ready").flatMap((h) => h.models || []));
  return (w.models || [w.model]).map((m) => ready.has(m) ? `${m}: served on its hosts`
    : (w.borrowing_models || []).includes(m) ? `${m}: on shared hosts meanwhile`
      : `${m}: refused (503) until its hosts are ready`);
}

function stateCell(w) {
  const several = (w.models || []).length > 1;
  const note = w.state === "preparing"
    ? (several ? null : w.borrowing ? "borrowing shared hosts" : `waiting — no shared host serves ${w.model}`)
    : w.state === "ending" ? "draining, key refused" : null;
  const perModel = w.state === "preparing" && several ? modelStartLines(w).map((line) => el("div", { class: "muted" }, line)) : [];
  let eta = null;
  if (w.state === "preparing" && w.plan && w.plan.minutes_to_serve) {
    const left = Math.max(0, Math.round((w.created_at + w.plan.minutes_to_serve * 60 - Date.now() / 1000) / 60));
    eta = left > 0 ? `serving in ~${left} min (planned)` : "due to serve any moment";
  }
  return el("td", {}, pill(w.state, WORKLOAD_STATES[w.state]),
    note ? el("div", { class: "muted" }, note) : null, ...perModel, eta ? el("div", { class: "muted" }, eta) : null);
}

function workloadRow(w) {
  const open = workloadDraft.open === w.name;
  const ready = w.hosts.filter((h) => h.state === "ready").length;
  const planned = Math.max(w.hosts_at_start || 0, w.hosts.length);
  const rows = [el("tr", {},
    el("td", {},
      el("button", { class: "link", "aria-expanded": open ? "true" : "false",
        onclick: () => { workloadDraft.open = open ? null : w.name; drawWorkloadTables(); } },
        el("span", { "aria-hidden": "true" }, open ? "▾ " : "▸ "), el("strong", {}, w.name)),
      el("div", { class: "muted mono" }, (w.models || [w.model]).join(", ")),
      (w.models || []).length > 1 ? el("div", { class: "muted" }, w.placement === "apart" ? "apart" : "together") : null,
      w.provisioner ? el("div", { class: "muted" }, `made by ${w.provisioner}`) : null),
    stateCell(w),
    w.state === "preparing" || w.state === "serving"
      ? el("td", { class: "num" }, `${ready} of ${planned}`, el("div", { class: "muted" }, "planned"))
      : el("td", { class: "num" }, w.hosts.length ? `${w.hosts.length} draining` : "—"),
    latencyCell(w),
    spendCell(w),
    timeLeftCell(w),
    el("td", {}, workloadActions(w), workloadDraft.rowErrors[w.name] ? el("div", { class: "error", role: "status" }, workloadDraft.rowErrors[w.name]) : null))];
  if (open) rows.push(el("tr", { class: "detail" }, el("td", { colspan: "7" }, workloadDetail(w))));
  return rows;
}

function workloadActions(w) {
  const live = w.state === "preparing" || w.state === "serving";
  if (!live) return el("span", { class: "muted" }, w.state === "ended" ? "" : "ending…");
  return el("div", { class: "row" },
    el("button", { class: "small", onclick: (e) => extendWorkload(e.target, w) }, "Extend…"),
    el("button", { class: "small", onclick: (e) => rotateWorkloadKey(e.target, w) }, "New key…"),
    el("button", { class: "small danger", onclick: (e) => endWorkload(e.target, w) }, "End…"));
}

// A workload's model volume, in one line (D139): where, how big, what it holds, what it costs.
const VOLUME_STATES = {
  filling: (v) => `being filled by ${v.filler}`,
  ready: (v) => `ready (${v.models.join(", ")})`,
  empty: () => "empty — the next host there fills it",
  stale: () => "holds an older build — the next host there fills the new one",
};

function volumeLine(v) {
  const state = (VOLUME_STATES[v.state] || (() => v.state || "?"))(v);
  return el("div", {}, `${v.connection} · ${v.location} · ${v.size_gb} GB · `,
    el("span", { class: v.state === "ready" ? "ok-text" : v.state === "filling" ? "" : "warn-text" }, state),
    el("span", { class: "muted" }, ` · ${rate(v.hourly)}, ${dollars(v.spent)} so far`));
}

function workloadDetail(w) {
  const answers = w.answers || {};
  const keyState = (k) => {
    if (w.state === "ending") return " — refused: the workload is ending";
    if (w.state !== "preparing" && w.state !== "serving") return " — refused: the workload has ended";
    if (k.not_after) return ` — works until ${until(k.not_after)}`;
    return " — in use";
  };
  const hostsLine = w.hosts.length ? null
    : w.state === "preparing" ? ((w.models || []).length > 1 ? `Its first hosts are being rented. ${modelStartLines(w).join("; ")}.`
      : w.borrowing ? "Its first hosts are being rented; meanwhile it is served on shared hosts."
        : `Its first hosts are being rented; until one is ready its requests are refused (no shared host serves ${w.model}).`)
    : w.state === "ending" ? "All its hosts are released; closing." : "No hosts.";
  return el("div", {},
    el("div", { class: "kv" },
      ...((w.models || []).length > 1 ? [
        el("div", { class: "k" }, "models"),
        el("div", {}, ...(w.targets || []).map((t) => {
          const share = (w.answers?.by_model || {})[t.model]?.cap_per_host;
          return el("div", {}, `${t.model}: ${t.parallel} answers at once in all, p95 within ${t.latency_s} s`
            + (share ? `; at most ${share} per host` : ""));
        })),
        el("div", { class: "k" }, "hosts"),
        el("div", {}, `${w.placement === "apart" ? "Apart" : "Together"} — `, ...(w.groups || []).map((g) => {
          const ready = w.hosts.filter((h) => h.state === "ready" && (h.models || []).join() === g.models.join()).length;
          return el("div", {}, `${g.models.join(" + ")}: ${ready} of ${g.hosts_at_start} ready, ${g.workers_per_host} answers at once each`);
        })),
      ] : [
        el("div", { class: "k" }, "answers at once"), el("div", {}, `${w.parallel}, on up to ${w.workers_per_host} per host`),
      ]),
      el("div", { class: "k" }, "machines"), el("div", {}, `${KIND_LABELS[w.kind] || w.kind} — ${KIND_HINTS[w.kind] || ""}`),
      el("div", { class: "k" }, "answers so far"),
      el("div", {}, `${answers.count || 0} served, ${answers.borrowed || 0} on shared hosts while starting, ${answers.refused || 0} refused`),
      el("div", { class: "k" }, "made by"), el("div", {}, w.provisioner ? `a program, with provisioning key ${w.provisioner}` : "an operator"),
      el("div", { class: "k" }, "lease"), el("div", { class: "mono" }, w.lease.lease_id),
      ...((w.volumes || []).length ? [el("div", { class: "k" }, "model volume"), el("div", {}, w.volumes.map(volumeLine))] : []),
      el("div", { class: "k" }, "keys"),
      el("div", {}, (w.keys || []).map((k) => el("div", {}, el("span", { class: "mono" }, k.key_id),
        el("span", { class: "muted" }, ` made ${until(k.created_at)}${keyState(k)}`))))),
    w.state === "ending" && w.hosts.length ? el("p", { class: "muted" }, `Draining: ${w.hosts.length} host${w.hosts.length === 1 ? "" : "s"} finishing their answers, then released.`) : null,
    w.hosts.length
      ? el("table", {}, el("thead", {}, el("tr", {}, ...["Host", "State", "Models", "Machine", "Kind", "Answers at once", "Price"].map((h) => el("th", {}, h)))),
          el("tbody", {}, w.hosts.map((h) => el("tr", {},
            el("td", {}, hostLink(h.host_id)), el("td", {}, pill(h.state)),
            el("td", {}, el("span", { class: "mono" }, (h.models || []).join(", ")),
              h.models_source && h.models_source !== "hub"
                ? el("div", { class: "muted" }, h.models_source === "volume" ? "copied from its model volume" : "copied from a sibling") : null),
            el("td", {}, h.hardware),
            el("td", {}, pricedPill(h), h.connection ? el("div", { class: "muted" }, h.connection) : null),
            el("td", { class: "num" }, h.workers),
            el("td", { class: "num" }, rate(h.hourly))))))
      : el("p", { class: "muted" }, hostsLine));
}

async function workloadAction(button, w, work) {
  delete workloadDraft.rowErrors[w.name];
  button.disabled = true;
  try { await work(); }
  catch (error) { workloadDraft.rowErrors[w.name] = error.message; }
  finally { button.disabled = false; await drawWorkloadTables(); }
}

async function extendWorkload(button, w) {
  const hours = el("input", { type: "number", min: "0.5", step: "0.5", placeholder: "none", style: "width:6rem" });
  const spend = el("input", { type: "number", min: "0.01", step: "0.01", value: (w.lease.max_spend ?? 0).toFixed(2), style: "width:7rem" });
  const perHour = w.hosts.reduce((sum, h) => sum + (h.hourly || 0), 0);
  const outcome = el("p", { class: "muted" });
  const describe = () => {
    const added = Number(hours.value) || 0;
    const cap = Number(spend.value) || w.lease.max_spend;
    const ends = (w.ends_at || 0) + added * 3600;
    let line = `Ends at ${until(ends)}${added ? ` (was ${until(w.ends_at)})` : ""}.`;
    if (perHour > 0) {
      const runsOut = Date.now() / 1000 + Math.max(0, cap - w.lease.spent) / perHour * 3600;
      line += ` At ${rate(perHour)} the budget of ${dollars(cap)} runs out at ${until(runsOut)}` + (runsOut < ends ? " — before its hours do." : ".");
    }
    outcome.textContent = line;
  };
  hours.addEventListener("input", describe); spend.addEventListener("input", describe); describe();
  const ok = await confirmAction({
    title: `Extend ${w.name}`,
    body: el("div", {},
      el("div", { class: "row" }, el("label", {}, "Add hours ", hours), el("label", {}, "Budget ($) ", spend)),
      outcome,
      el("p", { class: "muted" }, "Each raise is typed again on the next step.")),
    okLabel: "Next",
  });
  if (!ok) return;
  const body = {};
  const added = Number(hours.value);
  const cap = Number(spend.value);
  if (added > 0) {
    if (!(await confirmAction({ title: `Add ${added} hour${added === 1 ? "" : "s"} to ${w.name}?`, body: "More time is more spend.",
      retype: String(added), retypeLabel: `Type ${added} to add ${added} hour${added === 1 ? "" : "s"}`, okLabel: "Add hours" }))) return;
    body.hours = added; body.confirm_hours = added;
  }
  if (cap && cap !== w.lease.max_spend) {
    body.max_spend = cap;
    if (cap > w.lease.max_spend) {
      if (!(await confirmAction({ title: `Raise ${w.name}'s budget to ${dollars(cap)}?`, body: "Raising a dollar cap is loosening a limit.",
        retype: cap.toFixed(2), retypeLabel: `Type ${cap.toFixed(2)} to raise the budget`, okLabel: "Raise budget" }))) return;
      body.confirm = cap;
    }
  }
  if (!Object.keys(body).length) return;
  workloadAction(button, w, () => api.extendWorkload(w.name, body));
}

async function rotateWorkloadKey(button, w) {
  const ok = await confirmAction({ title: `A new key for ${w.name}?`, okLabel: "Make a new key",
    body: "The new key is shown once. The current one keeps working for the rotation grace, so the app can be switched without a gap." });
  if (!ok) return;
  workloadAction(button, w, async () => {
    const answer = await api.rotateWorkloadKey(w.name);
    workloadDraft.shown = { name: w.name, model: (w.models || [w.model]).join(", "), connection: answer.connection, rotated: true,
                            graceUntil: Date.now() / 1000 + answer.old_keys_valid_minutes * 60,
                            endsAt: w.ends_at, targets: w.targets || [] };
    render();
  });
}

async function endWorkload(button, w) {
  const ok = await confirmAction({
    title: `End ${w.name}?`,
    body: el("div", {},
      el("p", {}, `Spent so far ${dollars(w.lease.spent)} of ${dollars(w.lease.max_spend)}. Its lease closes, its ${w.hosts.length} host${w.hosts.length === 1 ? "" : "s"} finish what they are serving and are released, and its app gets 503 workload_ended from now.`),
      el("p", { class: "muted" }, "This cannot be undone: a new workload needs a new name.")),
    retype: w.name, retypeLabel: `Type ${w.name} to end it`, okLabel: "End workload",
  });
  if (ok) workloadAction(button, w, () => api.endWorkload(w.name));
}

// The tables only: refreshed every few seconds while this screen shows, and after each action,
// without touching the form or a key being copied.
async function drawWorkloadTables() {
  const holder = workloadDraft.holder;
  if (!holder || !holder.isConnected) return;
  let workloads;
  try { ({ workloads } = await api.workloads()); }
  catch (error) { holder.replaceChildren(el("p", { class: "error" }, error.message)); return; }
  const live = workloads.filter((w) => w.state !== "ended");
  const ended = workloads.filter((w) => w.state === "ended");
  const head = () => el("thead", {}, el("tr", {},
    el("th", {}, "Workload"), el("th", {}, "State"), el("th", { class: "num" }, "Hosts ready"),
    el("th", { class: "num" }, "Latency p95"), el("th", { class: "num" }, "Spent"), el("th", { class: "num" }, "Time left"), el("th", {}, "")));
  const endedOpen = holder.querySelector("details.ended")?.open;
  const hadLive = workloadDraft.hasLive;
  workloadDraft.hasLive = live.length > 0;
  if (hadLive !== workloadDraft.hasLive) drawWorkloadForm();
  holder.replaceChildren(...[
    el("h2", {}, `Active (${live.length})`),
    live.length ? el("table", { class: "workloads" }, head(), el("tbody", {}, live.flatMap(workloadRow)))
      : el("p", { class: "muted" }, "None. Apps with the pool's app key are served by the shared hosts."),
    ended.length ? el("details", { class: "ended", ...(endedOpen ? { open: true } : {}) }, el("summary", {}, `Ended (${ended.length})`),
      el("table", { class: "workloads" }, head(), el("tbody", {}, ended.flatMap(workloadRow)))) : null,
  ].filter(Boolean));  // replaceChildren would print a null; el() skips them, this does not
}

function drawWorkloadForm() {
  const d = workloadDraft;
  const holder = d.formHolder;
  if (!holder || !holder.isConnected || !state.status) return;
  const drafting = d.formOpen || d.name.trim() || d.plan || d.hasLive === false;
  if (drafting) {
    holder.replaceChildren(workloadForm(state.status));
    return;
  }
  if (d.hasLive === null) { holder.replaceChildren(); return; }  // not known yet: nothing to fold
  holder.replaceChildren(el("button", { class: "primary", onclick: () => { d.formOpen = true; drawWorkloadForm(); } },
    "New workload…"));
}

// --- applications that create their own workloads (D117) ---

const programDraft = { open: false, name: "", models: [], maxOpen: "1", perWorkload: "10", perDay: "20",
  hours: "8", certs: "optional", error: null, shown: null, holder: null };
const PROGRAM_NAME = /^[a-z0-9][a-z0-9-]{0,23}$/;

// Why "Create key" cannot be pressed yet, or null when it can.
function programBlocker(d) {
  const name = d.name.trim();
  if (!PROGRAM_NAME.test(name) || name.startsWith("rented")) return "The name is lower-case letters, digits and '-', up to 24";
  if (!d.models.length) return "Pick at least one model";
  if (!(Number(d.maxOpen) >= 1)) return "At least one workload at once";
  if (!(Number(d.perWorkload) > 0) || !(Number(d.perDay) > 0)) return "Budgets are above zero";
  if (Number(d.perWorkload) > Number(d.perDay)) return "One workload's budget cannot be above the day's";
  if (!(Number(d.hours) > 0)) return "Hours are above zero";
  return null;
}

function programKeyPanel(d, poolUrl) {
  const shown = d.shown;
  const heading = el("h2", { tabindex: "-1" }, `Provisioning key for ${shown.name} — give the application these`);
  const handoff = `GPM_URL=${poolUrl}\nGPM_PROVISIONING_KEY=${shown.key}`;
  const panel = el("div", { class: "panel key-panel", "aria-live": "polite" },
    heading,
    el("p", {}, el("strong", {}, "The key is shown this once."),
      " Only its hash is kept. It can create, use and end the application's own workloads within this grant, and nothing else."),
    el("div", { class: "kv" },
      el("div", { class: "k" }, "GPM_URL"), el("div", { class: "row" }, el("code", {}, poolUrl), copyButton(poolUrl)),
      el("div", { class: "k" }, "GPM_PROVISIONING_KEY"),
      el("div", { class: "row" }, el("code", { class: "secret" }, shown.key), copyButton(shown.key))),
    el("div", { class: "row" },
      copyButton(handoff, "Copy both for the application"),
      el("button", { onclick: async () => {
        if (await confirmAction({ title: "Put the key away?", body: "It will not be shown again. If it is lost, make a new key.",
          okLabel: "I have saved it" })) {
          d.shown = null; drawPrograms();
        }
      } }, "I have saved it")));
  if (!shown.focused) { shown.focused = true; setTimeout(() => { heading.scrollIntoView({ block: "start" }); heading.focus(); }, 0); }
  return panel;
}

function programRow(p, d) {
  const g = p.grant;
  const revoke = async (e, end) => {
    const open = p.open_now || 0;
    const ok = end
      ? await confirmAction({ title: `Revoke ${p.name} and end its workloads?`, okLabel: `Revoke and end ${open}`,
        body: el("div", {}, el("p", {}, `It can make no more workloads, and its ${open} open workload${open === 1 ? "" : "s"} end now: their hosts drain and are released.`),
          el("p", { class: "muted" }, "Answers in flight finish; new ones are refused.")),
        retype: p.name, retypeLabel: `Type ${p.name} to end them` })
      : await confirmAction({ title: `Revoke ${p.name}?`, okLabel: "Revoke",
        body: el("div", {}, el("p", {}, "It can make no more workloads, from now."),
          el("p", { class: "muted" }, open
            ? `Its ${open} open workload${open === 1 ? "" : "s"} keep running — and spending — until ended, out of hours or budget, or unused for their idle cutoff. "Revoke and end" stops them now.`
            : "It has no open workloads.")) });
    if (!ok) return;
    e.target.disabled = true;
    d.error = null;
    try { await api.revokeProvisioner(p.name, end); } catch (error) { d.error = `Revoking ${p.name}: ${error.message}`; }
    drawPrograms(); drawWorkloadTables();
  };
  return el("tr", {},
    el("td", {}, el("strong", {}, p.name), p.usable ? null : el("div", { class: "muted" }, p.revoked_at ? "revoked" : "expired")),
    el("td", {}, g.models.join(", ")),
    el("td", { class: "num" }, `${p.open_now || 0} of ${g.max_open}`),
    el("td", { class: "num" }, dollars(g.max_spend)),
    el("td", { class: "num" }, `${dollars(p.committed_today)} of ${dollars(g.max_spend_per_day)}`),
    el("td", { class: "num" }, `${g.max_hours} h`),
    el("td", {}, g.certs === "required" ? "key and certificate" : "key only"),
    el("td", {}, p.usable ? el("div", { class: "row" },
      el("button", { class: "small danger", onclick: (e) => revoke(e, false) }, "Revoke…"),
      p.open_now ? el("button", { class: "small danger", onclick: (e) => revoke(e, true) }, `Revoke and end ${p.open_now}…`) : null) : null));
}

function programTable(list, d) {
  return el("table", { class: "programs-table" }, el("thead", {}, el("tr", {},
    ...["Application", "Models", "Open now", "Per workload", "Committed today", "Hours", "Access", ""]
      .map((h, i) => el("th", i >= 2 && i <= 5 ? { class: "num" } : {}, h)))),
  el("tbody", {}, list.map((p) => programRow(p, d))));
}

function programForm(d, models) {
  const create = el("button", { class: "primary" }, "Create key");
  const sync = () => { const why = programBlocker(d); create.disabled = !!why; create.title = why || ""; };
  const field = (label, key, attrs, hint) => el("label", { class: "field" }, el("span", {}, label),
    el("input", { ...attrs, value: d[key], oninput: (e) => { d[key] = e.target.value; sync(); } }),
    hint ? el("small", { class: "muted" }, hint) : null);
  create.onclick = async (e) => {
    const body = { name: d.name.trim(), models: d.models, max_open: Number(d.maxOpen), max_spend: Number(d.perWorkload),
      max_spend_per_day: Number(d.perDay), max_hours: Number(d.hours), certs: d.certs };
    const ok = await confirmAction({ title: `A provisioning key for ${body.name}?`, okLabel: "Create key",
      body: `It may commit up to ${dollars(body.max_spend_per_day)} a day, in workloads of up to ${dollars(body.max_spend)} each. Type the daily budget to confirm.`,
      retype: body.max_spend_per_day.toFixed(2), retypeLabel: `Type ${body.max_spend_per_day.toFixed(2)} to allow it` });
    if (!ok) return;
    e.target.disabled = true;
    try {
      const made = await api.createProvisioner(body);
      Object.assign(d, { shown: { name: body.name, key: made.key }, open: false, name: "", models: [], error: null });
    } catch (error) { d.error = error.message; }
    drawPrograms();
  };
  sync();
  return el("div", { class: "panel" },
    el("h3", {}, "New provisioning key"),
    el("p", { class: "muted" }, "Lets one application create, use and end its own workloads through the SDK, within these limits. The pool's own daily cap still bounds all of them together."),
    el("div", { class: "form-grid" },
      field("Application", "name", { type: "text", maxlength: "24", placeholder: "e.g. nightly-evals" },
        "lower-case letters, digits and '-'"),
      el("fieldset", { class: "field" }, el("legend", {}, "Models"),
        ...models.map((m) => el("label", { class: "pick" },
          el("input", { type: "checkbox", value: m, ...(d.models.includes(m) ? { checked: true } : {}),
            onchange: (e) => {
              d.models = e.target.checked ? [...d.models, m] : d.models.filter((x) => x !== m);
              sync();
            } }),
          el("span", { class: "mono" }, m)))),
      field("Workloads at once", "maxOpen", { type: "number", min: "1", step: "1" }),
      field("Budget per workload ($)", "perWorkload", { type: "number", min: "0.01", step: "0.01" }),
      field("Budget per day ($)", "perDay", { type: "number", min: "0.01", step: "0.01" }, "all its workloads together, over any 24 hours"),
      field("Most hours", "hours", { type: "number", min: "0.5", step: "0.5" }, "per workload"),
      el("label", { class: "field" }, el("span", {}, "Client certificate"),
        el("select", { onchange: (e) => { d.certs = e.target.value; } },
          ...[["optional", "not needed"], ["required", "required"]].map(([v, t]) => el("option", { value: v, ...(d.certs === v ? { selected: true } : {}) }, t))),
        el("small", { class: "muted" }, "required: the application must also present a certificate the pool signed for the workload"))),
    el("div", { class: "row" },
      create,
      el("button", { onclick: () => { d.open = false; d.error = null; drawPrograms(); } }, "Cancel")));
}

async function drawPrograms() {
  const d = programDraft;
  const holder = d.holder;
  if (!holder || !holder.isConnected) return;
  let answer;
  try { answer = await api.provisioners(); }
  catch (error) { holder.replaceChildren(el("p", { class: "error" }, error.message)); return; }
  const provisioners = answer.provisioners || [];
  const status = state.status || {};
  const models = [...new Set([...(status.workload_models || []), ...(status.model_set || [])])];
  const usable = provisioners.filter((p) => p.usable);
  const gone = provisioners.filter((p) => !p.usable);
  const goneOpen = holder.querySelector("details.gone")?.open;
  holder.replaceChildren(...[
    el("h2", {}, `Applications that create workloads (${usable.length})`),
    el("p", { class: "muted" }, "An application holding a provisioning key creates its own workloads through the SDK, within its grant.",
      answer.max_spend_per_day ? ` All of them together commit at most ${dollars(answer.max_spend_per_day)} a day.` : ""),
    d.error ? el("p", { class: "error", role: "status" }, d.error) : null,
    d.shown ? programKeyPanel(d, answer.pool_url || "") : null,
    usable.length ? programTable(usable, d) : el("p", { class: "muted" }, "None: only operators create workloads."),
    gone.length ? el("details", { class: "gone", ...(goneOpen ? { open: true } : {}) },
      el("summary", {}, `Revoked or expired (${gone.length})`), programTable(gone, d)) : null,
    d.open ? programForm(d, models) : el("button", { onclick: () => { d.open = true; drawPrograms(); } }, "New provisioning key…"),
  ].filter(Boolean));
}

screens.workloads = (status) => {
  const holder = el("div", {}, el("p", { class: "muted" }, "Loading workloads…"));
  workloadDraft.holder = holder;
  workloadDraft.formHolder = el("div", { class: "form-holder" });
  clearInterval(workloadDraft.timer);
  workloadDraft.timer = setInterval(() => {
    if (state.screen !== "workloads") { clearInterval(workloadDraft.timer); return; }
    // Not under the operator: a dialog open, focus in the table, or text being selected.
    if (document.querySelector("dialog[open]") || workloadDraft.holder?.contains(document.activeElement)
        || String(getSelection()).length) return;
    drawWorkloadTables();
  }, 5000);
  programDraft.holder = el("div", { class: "programs" });
  setTimeout(() => { drawWorkloadTables(); drawWorkloadForm(); drawPrograms(); }, 0);
  return [el("h1", {}, "Workloads"), keyPanel(), holder, workloadDraft.formHolder, programDraft.holder];
};

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
                    change.needs_restart ? el("span", { class: "pill warn" }, ` needs a ${change.restarts || "router"} restart `) : null,
                    change.refused ? el("div", { class: "error" }, `Refused: ${change.refused}`) : null)))
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
  // `#rented/engine`: the screen, then which of its tabs.
  const [name, ...rest] = (location.hash.replace("#", "") || "overview").split("/");
  state.screen = name;
  state.sub = rest.join("/") || null;
  for (const link of document.querySelectorAll("#nav a")) {
    link.classList.toggle("active", link.getAttribute("href") === `#${name}`);
  }
  const main = document.getElementById("screen");
  // The same screen drawn again — a status update every few seconds, or after an action — keeps
  // the operator's place: no "Loading…" in between and the scroll where it was. Found by the
  // owner: the page jumped back to the top every few seconds while scrolled down.
  const where = `${name}/${state.sub || ""}`;
  const again = state.drawn === where;
  state.drawn = where;
  // Boxes that scroll on their own — the live feed, a wide table's panel — keep theirs too: the
  // overview redraws on every decision, and the feed went back to its top each time (the owner).
  const SCROLLERS = ".feed, .panel, pre, textarea";
  const draw = (...children) => {
    const y = window.scrollY;
    const inner = again ? [...main.querySelectorAll(SCROLLERS)].map((box) => [box.scrollTop, box.scrollLeft]) : [];
    main.replaceChildren(...children);
    if (!again) return;
    [...main.querySelectorAll(SCROLLERS)].forEach((box, i) => {
      if (inner[i] && (inner[i][0] || inner[i][1])) { box.scrollTop = inner[i][0]; box.scrollLeft = inner[i][1]; }
    });
    window.scrollTo(0, y);
  };
  try {
    const pending = (screens[name] || screens.overview)(state.status);
    if (pending instanceof Promise && !again) {
      // A screen that has to ask the provider takes seconds. Say so, rather than leaving the
      // previous screen on the page where it reads as this one's answer.
      main.replaceChildren(el("p", { class: "muted" }, "Loading…"));
    }
    const parts = await pending;
    if (state.screen !== name || state.drawn !== where) return;  // the operator moved on while we were waiting
    draw(...[parts].flat().filter(Boolean));
  } catch (error) {
    if (state.screen !== name) return;
    draw(el("p", { class: "error" }, error.message));
  }
}

// The release this page's code came from. A page keeps its code until it is reloaded, so after a
// deploy a forgotten tab would go on running the old one — found live: a tab from before the
// fix spent the provider's daily search quota every day (D122). On a newer release it reloads
// itself, unless the operator is in the middle of something; then it says so instead.
function operatorIsBusy() {
  const active = document.activeElement;
  return Boolean(
    workloadDraft.shown || programDraft.shown || document.querySelector("dialog[open]")
    || (active && ["INPUT", "TEXTAREA", "SELECT"].includes(active.tagName))
    || workloadDraft.plan || (workloadDraft.name || "").trim() || programDraft.open);
}

function checkRelease(status) {
  const running = status && status.server_version;
  if (!running) return;
  if (state.loadedRelease == null) { state.loadedRelease = running; return; }
  if (running === state.loadedRelease) return;
  if (!operatorIsBusy()) { location.reload(); return; }
  if (document.getElementById("release-banner")) return;
  const banner = el("div", { id: "release-banner", class: "panel warn-text", role: "status" },
    `The pool now runs ${running}; this page is from ${state.loadedRelease}. Reload it when you are done here. `,
    el("button", { class: "small", onclick: () => location.reload() }, "Reload now"));
  document.getElementById("screen").before(banner);
}

async function refresh() {
  state.status = await api.status();
  checkRelease(state.status);
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
    checkRelease(payload);
    document.getElementById("pool-name").textContent = payload.pool;
    if (["overview", "hosts", "models"].includes(state.screen)) render();
    else if (state.screen === "rented" && state.sub === "providers" && !operatorIsBusy()) render();
  }
}

async function start() {
  document.getElementById("sign-out").hidden = false;
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

// Where the admin key is kept between reloads (D137): this tab, or this browser until Sign out.
const KEY_SLOT = "gpm.admin-key";
const keptKey = {
  read() {
    try { return localStorage.getItem(KEY_SLOT) || sessionStorage.getItem(KEY_SLOT); } catch { return null; }
  },
  write(key, browser) {
    try {
      this.forget();
      (browser ? localStorage : sessionStorage).setItem(KEY_SLOT, key);
    } catch { /* storage refused: the key lives in this page only */ }
  },
  forget() {
    try { localStorage.removeItem(KEY_SLOT); sessionStorage.removeItem(KEY_SLOT); } catch { /* nothing kept */ }
  },
};

const keyDialog = document.getElementById("key-dialog");
document.getElementById("key-form").onsubmit = async (event) => {
  event.preventDefault();
  ADMIN_KEY = document.getElementById("key-input").value.trim();
  const error = document.getElementById("key-error");
  try {
    await api.status();
    keptKey.write(ADMIN_KEY, document.getElementById("key-remember").checked);
    document.getElementById("key-input").value = "";
    keyDialog.close();
    await start();
  } catch (problem) {
    ADMIN_KEY = null;
    error.textContent = problem.message;
    error.hidden = false;
  }
};
document.getElementById("sign-out").onclick = () => {
  keptKey.forget();
  ADMIN_KEY = null;
  location.reload();
};

// A key kept from before opens the console straight away; one the pool no longer takes is
// forgotten, and the key is asked for again.
(async () => {
  const kept = keptKey.read();
  if (kept) {
    ADMIN_KEY = kept;
    try {
      await api.status();
      await start();
      return;
    } catch {
      ADMIN_KEY = null;
      keptKey.forget();
    }
  }
  keyDialog.showModal();
})();
