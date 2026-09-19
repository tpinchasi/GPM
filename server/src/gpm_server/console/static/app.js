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
    throw new Error(detail);
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
      ? call("POST", `/pool/market/preview?hours=${hours}`, policy)
      : call("GET", `/pool/market/preview?hours=${hours}`),
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
    tiers.get(20).push({ ...host, kind: "rented-interruptible", rented: true });
  }
  const ready = status.hosts.filter((h) => h.state === "ready").length;
  const workers = status.hosts.reduce((n, h) => n + (h.workers || 0), 0);
  const busy = status.hosts.reduce((n, h) => n + (h.busy || 0), 0);
  const burn = status.rented.reduce((n, h) => n + (h.bid_hourly || 0), 0);

  return [
    el("h1", {}, "Overview"),
    el("div", { class: "grid" },
      el("div", { class: "panel" }, el("h2", {}, "Capacity"),
        el("div", { class: "stat" }, `${busy} / ${workers}`),
        el("div", { class: "muted" }, `workers busy · ${ready} host(s) ready`)),
      el("div", { class: "panel" }, el("h2", {}, "Rented"),
        el("div", { class: "stat" }, status.rented.length),
        el("div", { class: "muted" }, `burning ${rate(burn)} · cap ${rate(status.limits.max_hourly_burn)}`)),
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
            el("td", { class: "mono" }, host.host_id),
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
          el("td", {}, el("button", { class: "small", onclick: (e) => run(e.target, () => api.closeLease(lease.lease_id)) }, "Close")));
      })))
  : el("p", { class: "muted" }, "No open lease, so nothing can be rented.");

const feed = () => el("div", { class: "panel feed" }, state.events.slice(0, 40).map((event) =>
  el("div", { class: "ev" },
    el("span", { class: "k" }, `${clock(event.ts)}  ${event.kind}  `),
    event.summary)));

screens.hosts = (status) => [
  el("h1", {}, "Hosts"),
  el("p", { class: "muted" }, "Configured hosts. Add, edit and disable are configuration changes, so they go through plan on the Configuration screen."),
  el("table", {},
    el("thead", {}, el("tr", {},
      el("th", {}, "Host"), el("th", {}, "Kind"), el("th", {}, "Transport"), el("th", {}, "State"),
      el("th", { class: "num" }, "Workers"), el("th", {}, "Capabilities"), el("th", {}, "Residency"), el("th", {}, "Tunnel"), el("th", { class: "num" }, "Served"))),
    el("tbody", {}, status.hosts.map((host) => el("tr", {},
      el("td", { class: "mono" }, host.host_id),
      el("td", {}, host.kind),
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
          el("div", { class: "k" }, "max rented hosts"), el("div", {}, status.limits.max_rented_hosts),
          el("div", { class: "k" }, "max hourly burn"), el("div", {}, rate(status.limits.max_hourly_burn))),
        el("p", { class: "muted" }, "Raising either is a configuration change that must be retyped to confirm.")),
      preparePanel(),
    ),
    el("h2", {}, "Live market — the real offer pipeline, read-only"),
    marketPanel(market),
    el("h2", {}, "Rented and parked hosts"),
    status.rented.length ? el("table", {},
      el("thead", {}, el("tr", {},
        el("th", {}, "Host"), el("th", {}, "State"), el("th", {}, "Machine"), el("th", { class: "num" }, "Bid"),
        el("th", { class: "num" }, "Storage"), el("th", { class: "num" }, "Held"), el("th", { class: "num" }, "Spend"), el("th", {}, ""))),
      el("tbody", {}, status.rented.map((host) => el("tr", {},
        el("td", { class: "mono" }, host.host_id),
        el("td", {}, pill(host.state)),
        el("td", { class: "muted" }, `${host.machine} · ${host.hardware || ""}`),
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

const marketPanel = (market) => {
  if (market.error) return el("p", { class: "error" }, market.error);
  return el("div", { class: "grid" },
    el("div", { class: "panel" },
      el("div", { class: "stat" }, `${market.passed} pass · ${market.rejected} rejected`),
      el("div", { class: "muted" }, `${market.seen} offers seen through your policy`),
      el("h2", {}, "Rejected, by reason"),
      el("table", {}, el("tbody", {}, Object.entries(market.rejected_by_reason || {}).map(([reason, count]) =>
        el("tr", {}, el("td", { class: "num" }, count), el("td", { class: "muted" }, reason)))))),
    el("div", { class: "panel" }, el("h2", {}, "Best offers"),
      el("table", {},
        el("thead", {}, el("tr", {},
          el("th", {}, "Hardware"), el("th", { class: "num" }, "Floor"), el("th", { class: "num" }, "Would bid"),
          el("th", { class: "num" }, "On-demand"), el("th", { class: "num" }, "$/GB"), el("th", { class: "num" }, "Score"))),
        el("tbody", {}, (market.best || []).map((offer, index) => el("tr", {},
          el("td", {}, index === 0 ? el("strong", {}, offer.hardware) : offer.hardware,
            el("div", { class: "muted mono" }, `${offer.machine} · ${offer.gpu_memory_gb}GB · ${offer.download_mbps}Mbps`)),
          el("td", { class: "num" }, rate(offer.floor)),
          el("td", { class: "num" }, el("strong", {}, rate(offer.would_bid))),
          el("td", { class: "num" }, rate(offer.on_demand)),
          el("td", { class: "num" }, `$${Number(offer.download_per_gb).toFixed(4)}`),
          el("td", { class: "num" }, Math.round(offer.score))))))));
};

function preparePanel() {
  const spend = el("input", { type: "number", step: "0.01", value: "1.00", style: "width:6rem" });
  const hours = el("input", { type: "number", step: "0.5", value: "1", style: "width:5rem" });
  const when = el("select", {}, el("option", { value: "join" }, "join the pool"), el("option", { value: "park" }, "park it"), el("option", { value: "destroy" }, "destroy"));
  return el("div", { class: "panel" }, el("h2", {}, "Prepare a host"),
    el("p", { class: "muted" }, "Its own small lease: rent, load the model set, verify, then join, park or destroy. Borrows no authority from any other lease."),
    el("label", {}, "Dollar cap ", spend),
    el("label", {}, "Time limit (hours) ", hours),
    el("label", {}, "When ready ", when),
    el("div", { class: "row" }, el("button", { class: "primary", onclick: async (e) => {
      const worst = el("div", {}, el("p", {}, `This spends money. Worst case: ${money(Number(spend.value))} over ${hours.value}h, one host.`),
        el("p", { class: "muted" }, "The pool bids on the best offer its policy allows, and stops at the cap less the safety margin."));
      if (!(await confirmAction({ title: "Prepare a host?", body: worst }))) return;
      run(e.target, () => api.prepare({ max_spend: Number(spend.value), max_hours: Number(hours.value), when_ready: when.value }));
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
    el("h2", {}, "All leases"),
    el("table", {},
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
        el("td", {}, lease.state === "open"
          ? el("div", { class: "row" },
              el("button", { class: "small", onclick: async (e) => {
                const value = prompt("Tighten the dollar cap to:", String(lease.max_spend));
                if (value === null) return;
                run(e.target, () => api.tightenLease(lease.lease_id, { max_spend: Number(value) }));
              } }, "Tighten"),
              el("button", { class: "small", onclick: (e) => run(e.target, () => api.closeLease(lease.lease_id)) }, "Close"))
          : null))))),
  ];
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
