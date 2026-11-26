"use strict";
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const HUMAN = "It was on a dreary night of November that I beheld the accomplishment of my toils. With an anxiety that almost amounted to agony, I collected the instruments of life around me, that I might infuse a spark of being into the lifeless thing that lay at my feet. It was already one in the morning; the rain pattered dismally against the panes, and my candle was nearly burnt out, when, by the glimmer of the half-extinguished light, I saw the dull yellow eye of the creature open; it breathed hard, and a convulsive motion agitated its limbs.";

function settings() {
  return {scheme: $("#scheme").value, gamma: parseFloat($("#gamma").value), delta: parseFloat($("#delta").value)};
}
async function post(url, data) {
  const r = await fetch(url, {method: "POST", headers: {"Content-Type": "application/json", "X-Wmlab": "1"}, body: JSON.stringify(data)});
  const j = await r.json().catch(() => ({error: "bad response"}));
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}
function busy(btn, on) { btn.disabled = on; }
function detCard(d, title) {
  const cls = d.detected ? "yes" : (d.verdict.includes("short") ? "short" : "no");
  const toks = (d.flags || []).map((f) => `<span class="tok ${f.status}" title="${f.status}${f.score != null ? " " + f.score : ""}">${esc(f.token)}</span>`).join(" ");
  const pct = Math.max(0, Math.min(100, (d.z / 12) * 100));
  return `<div class="card">${title ? `<div class="legend">${esc(title)}</div>` : ""}
    <div class="verdict ${cls}">${esc(d.verdict)}</div>
    <div class="meter"><i data-w="${pct.toFixed(1)}"></i></div>
    <div class="stats"><span class="stat">z = <b>${d.z.toFixed(2)}</b></span><span class="stat">p = <b>${d.p_value.toExponential(2)}</b></span>
    <span class="stat">green <b>${d.num_green}/${d.num_scored}</b> (${Math.round(d.green_fraction * 100)}%)</span><span class="stat">scheme <b>${esc(d.scheme)}</b></span></div>
    ${toks ? `<div class="tokens">${toks}</div><div class="legend">green = word in the keyed green list · red = not green · grey = context / repeated pair (not scored)</div>` : ""}
    <div class="caution">A detected mark is statistical evidence about one key. It is not proof of authorship, and a missing mark does not prove a human wrote the text.</div></div>`;
}
// CSP forbids inline style attributes, so bar widths are set through the CSSOM
new MutationObserver(() => document.querySelectorAll("[data-w]").forEach((e) => { e.style.width = e.dataset.w + "%"; e.removeAttribute("data-w"); }))
  .observe(document.body, {childList: true, subtree: true});
function fail(el, e) { el.innerHTML = `<div class="card err">${esc(e.message)}</div>`; }

document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === b));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === b.dataset.tab));
}));

$("#gen-go").addEventListener("click", async (ev) => {
  const out = $("#gen-out"); busy(ev.target, true);
  try {
    const j = await post("/api/generate", {...settings(), prompt: $("#gen-prompt").value, tokens: parseInt($("#gen-tokens").value, 10), watermark: $("#gen-wm").checked, seed: 7});
    out.innerHTML = `<div class="card"><div class="legend">${j.watermarked ? "watermarked output" : "plain output (no watermark)"}</div><p>${esc(j.text)}</p></div>` + detCard(j.detection, "detection on the generated text");
    $("#det-text").value = j.text; $("#att-text").value = j.text; $("#prov-text").value = j.text;
  } catch (e) { fail(out, e); } finally { busy(ev.target, false); }
});
$("#det-human").addEventListener("click", () => { $("#det-text").value = HUMAN; });
$("#det-go").addEventListener("click", async (ev) => {
  const out = $("#det-out"); busy(ev.target, true);
  try { const j = await post("/api/detect", {...settings(), text: $("#det-text").value}); out.innerHTML = detCard(j.detection); }
  catch (e) { fail(out, e); } finally { busy(ev.target, false); }
});
$("#att-go").addEventListener("click", async (ev) => {
  const out = $("#att-out"); busy(ev.target, true);
  try {
    const j = await post("/api/attack", {...settings(), text: $("#att-text").value, type: $("#att-type").value, rate: parseFloat($("#att-rate").value)});
    const b = j.before, a = j.after, raw = j.after_without_normalisation;
    out.innerHTML = `<div class="cmp"><div class="card"><div class="legend">before</div><div class="verdict ${b.detected ? "yes" : "no"}">z = ${b.z.toFixed(2)}</div>${esc(b.verdict)}</div>
      <div class="card"><div class="legend">after ${esc($("#att-type").value)} @ ${$("#att-rate").value}</div><div class="verdict ${a.detected ? "yes" : "no"}">z = ${a.z.toFixed(2)}</div>${esc(a.verdict)}
      <div class="legend">without normalisation: z = ${raw.z.toFixed(2)}</div></div></div>
      <div class="card"><div class="legend">attacked text</div><p>${esc(j.text)}</p></div>` + detCard(a, "detection after the attack");
  } catch (e) { fail(out, e); } finally { busy(ev.target, false); }
});
function verCard(v) {
  const cls = v.status === "valid" ? "valid" : (v.status === "valid_untrusted" ? "valid_untrusted" : "bad");
  return `<div class="card"><div class="status ${cls}">${esc(v.status.toUpperCase())}</div>
    ${v.signer ? `<div class="legend">signer key id ${esc(v.signer)}${v.source_type ? " · source " + esc(v.source_type) : ""}</div>` : ""}
    <ul class="checks">${v.checks.map((c) => `<li>${esc(c)}</li>`).join("")}</ul></div>`;
}
$("#prov-sign").addEventListener("click", async (ev) => {
  const out = $("#prov-out"); busy(ev.target, true);
  try {
    const j = await post("/api/sign", {text: $("#prov-text").value, source: $("#prov-source").value, watermarked: $("#prov-wm").checked, scheme: settings().scheme, title: "wmlab web demo"});
    $("#prov-manifest").value = JSON.stringify(j.manifest, null, 2);
    out.innerHTML = `<div class="card"><div class="status valid">SIGNED</div><div class="legend">Ed25519 key ${esc(j.manifest.signature.key_id)} · sha256 ${esc(j.manifest.hard_binding.hash.slice(0, 16))}…</div></div>`;
  } catch (e) { fail(out, e); } finally { busy(ev.target, false); }
});
$("#prov-verify").addEventListener("click", async (ev) => {
  const out = $("#prov-out"); busy(ev.target, true);
  try { const j = await post("/api/verify", {text: $("#prov-text").value, manifest: $("#prov-manifest").value || null}); out.innerHTML = verCard(j.verification); }
  catch (e) { fail(out, e); } finally { busy(ev.target, false); }
});
$("#prov-tamper").addEventListener("click", () => {
  const t = $("#prov-text").value; const w = t.split(" ");
  if (w.length > 3) { w[3] = w[3] === "never" ? "always" : "never"; $("#prov-text").value = w.join(" "); }
});
