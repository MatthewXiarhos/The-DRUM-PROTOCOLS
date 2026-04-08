(function () {
  "use strict";

  var WORKER_URL = "https://tdp-chat.the-drum-protocols.workers.dev";
  var TURNSTILE_KEY = "0x4AAAAAAC175HN3K_6IV1fG";
  var EXCLUDE_PATHS = ["/drum-protocols-admin-v5.html", "/admin"];

  var path = window.location.pathname;
  for (var i = 0; i < EXCLUDE_PATHS.length; i++) {
    if (path.indexOf(EXCLUDE_PATHS[i]) !== -1) return;
  }

  var C = {
    gold: "#D4A017",
    goldDim: "#B8860B",
    goldBorder: "rgba(184,134,11,0.35)",
    goldFaint: "rgba(184,134,11,0.08)",
    goldGlow: "rgba(212,160,23,0.18)",
    bg: "#0d1720",
    surface: "#131e28",
    surface2: "#1a2633",
    surface3: "#243345",
    text: "#e8e4dc",
    text2: "#9aabb8",
    text3: "#607080",
    border: "rgba(255,255,255,0.06)",
    borderMid: "rgba(255,255,255,0.10)",
    healing: "#52B788",
    danger: "#c0392b",
    mono: "'DM Mono', monospace",
    sans: "'DM Sans', system-ui, sans-serif"
  };

  var SPHERE_LOGO_SM =
    '<svg viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg" style="width:20px;height:20px">' +
    '<ellipse cx="10" cy="10" rx="8.5" ry="5" stroke="' + C.gold + '" stroke-width="1.2"/>' +
    '<line x1="1.5" y1="10" x2="18.5" y2="10" stroke="' + C.gold + '" stroke-width="1"/>' +
    '<path d="M5 10 Q6.5 6.5 10 6.5 Q13.5 6.5 15 10 Q13.5 13.5 10 13.5 Q6.5 13.5 5 10Z" stroke="' + C.goldDim + '" stroke-width="0.8" fill="none"/>' +
    '</svg>';

  var SPHERE_LOGO_LG =
    '<svg viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg" style="width:28px;height:28px">' +
    '<ellipse cx="14" cy="14" rx="12" ry="7" stroke="' + C.gold + '" stroke-width="1.4"/>' +
    '<line x1="2" y1="14" x2="26" y2="14" stroke="' + C.gold + '" stroke-width="1"/>' +
    '<path d="M7 14 Q9 9 14 9 Q19 9 21 14 Q19 19 14 19 Q9 19 7 14Z" stroke="' + C.goldDim + '" stroke-width="1" fill="none"/>' +
    '<circle cx="14" cy="14" r="1.5" fill="' + C.gold + '"/>' +
    '</svg>';

  var FAB_LOGO =
    '<svg viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg" style="width:26px;height:26px">' +
    '<ellipse cx="16" cy="16" rx="13" ry="8" stroke="' + C.bg + '" stroke-width="1.5"/>' +
    '<line x1="3" y1="16" x2="29" y2="16" stroke="' + C.bg + '" stroke-width="1.2"/>' +
    '<path d="M8 16 Q10.5 10 16 10 Q21.5 10 24 16 Q21.5 22 16 22 Q10.5 22 8 16Z" stroke="' + C.bg + '" stroke-width="1.2" fill="none"/>' +
    '<circle cx="16" cy="16" r="2" fill="' + C.bg + '"/>' +
    '</svg>';

  var CSS = [
    "#tdp-fab{position:fixed;bottom:24px;right:24px;z-index:9998;width:52px;height:52px;border-radius:50%;background:linear-gradient(135deg," + C.gold + "," + C.goldDim + ");border:none;cursor:pointer;box-shadow:0 4px 20px " + C.goldGlow + ",0 2px 8px rgba(0,0,0,0.4);display:flex;align-items:center;justify-content:center;transition:transform .15s,box-shadow .15s;}",
    "#tdp-fab:hover{transform:scale(1.07);box-shadow:0 6px 28px " + C.goldGlow + ",0 2px 10px rgba(0,0,0,0.5);}",
    "#tdp-panel{position:fixed;bottom:88px;right:24px;z-index:9999;width:410px;max-width:calc(100vw - 32px);height:470px;max-height:calc(100vh - 110px);background:" + C.bg + ";border:1px solid " + C.goldBorder + ";border-radius:16px;display:flex;flex-direction:column;overflow:hidden;font-family:" + C.sans + ";font-size:14px;box-shadow:0 20px 60px rgba(0,0,0,.7),0 0 0 1px " + C.goldFaint + ";transform:translateY(14px) scale(.97);opacity:0;pointer-events:none;transition:transform .2s cubic-bezier(.4,0,.2,1),opacity .2s;}",
    "#tdp-panel.tdp-open{transform:translateY(0) scale(1);opacity:1;pointer-events:all;}",
    "#tdp-hdr{flex-shrink:0;display:flex;align-items:center;justify-content:space-between;padding:11px 14px;border-bottom:1px solid " + C.goldBorder + ";background:" + C.surface + ";border-top:2px solid " + C.gold + ";}",
    "#tdp-hdr-l{display:flex;align-items:center;gap:10px;}",
    "#tdp-title{font-size:13px;font-weight:600;color:" + C.text + ";letter-spacing:.01em;font-family:" + C.sans + ";}",
    "#tdp-sub{font-size:10px;color:" + C.text3 + ";font-family:" + C.mono + ";margin-top:1px;letter-spacing:.03em;}",
    "#tdp-close{width:26px;height:26px;border-radius:50%;border:1px solid " + C.border + ";background:transparent;color:" + C.text3 + ";cursor:pointer;font-size:16px;display:flex;align-items:center;justify-content:center;transition:all .12s;}",
    "#tdp-close:hover{border-color:" + C.goldBorder + ";color:" + C.gold + ";}",
    "#tdp-tier{flex-shrink:0;padding:6px 13px;background:" + C.bg + ";border-bottom:1px solid " + C.border + ";display:none;align-items:center;gap:8px;}",
    "#tdp-tier.tdp-vis{display:flex;}",
    ".tdp-tbadge{font-family:" + C.mono + ";font-size:9px;padding:2px 7px;border-radius:20px;border:1px solid " + C.goldBorder + ";letter-spacing:.03em;color:" + C.gold + ";background:" + C.goldFaint + ";}",
    ".tdp-tsrc{font-family:" + C.mono + ";font-size:9px;color:" + C.text3 + ";white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}",
    "#tdp-msgs{flex:1;overflow-y:auto;padding:13px;display:flex;flex-direction:column;gap:11px;scroll-behavior:smooth;}",
    "#tdp-msgs::-webkit-scrollbar{width:3px;}",
    "#tdp-msgs::-webkit-scrollbar-thumb{background:rgba(184,134,11,0.25);border-radius:2px;}",
    ".tdp-mu{align-self:flex-end;max-width:78%;background:" + C.surface3 + ";border:1px solid " + C.borderMid + ";border-radius:12px 12px 3px 12px;padding:9px 13px;font-size:13px;line-height:1.6;color:" + C.text + ";white-space:pre-wrap;}",
    ".tdp-ma{align-self:flex-start;max-width:88%;}",
    ".tdp-mah{display:flex;align-items:center;gap:7px;margin-bottom:4px;}",
    ".tdp-man{font-family:" + C.mono + ";font-size:10px;color:" + C.gold + ";letter-spacing:.05em;}",
    ".tdp-mam{font-family:" + C.mono + ";font-size:10px;color:" + C.text3 + ";}",
    ".tdp-mab{background:" + C.surface + ";border:1px solid " + C.borderMid + ";border-radius:3px 12px 12px 12px;padding:11px 14px;font-size:13px;line-height:1.7;color:" + C.text + ";white-space:pre-wrap;}",
    ".tdp-mab.tdp-welcome{border-color:" + C.goldBorder + ";background:" + C.goldFaint + ";}",
    ".tdp-src{margin-top:5px;display:flex;flex-wrap:wrap;gap:4px;}",
    ".tdp-srctag{font-family:" + C.mono + ";font-size:9px;padding:2px 7px;border-radius:20px;border:1px solid " + C.goldBorder + ";color:" + C.gold + ";background:" + C.goldFaint + ";}",
    ".tdp-fb{display:flex;gap:7px;margin-top:5px;}",
    ".tdp-fbb{font-size:13px;background:transparent;border:1px solid " + C.border + ";border-radius:6px;padding:3px 7px;cursor:pointer;color:" + C.text3 + ";transition:all .12s;}",
    ".tdp-fbb:hover{background:" + C.surface2 + ";color:" + C.text + ";border-color:" + C.borderMid + ";}",
    ".tdp-fbb.tdp-voted{border-color:" + C.goldBorder + ";color:" + C.gold + ";}",
    ".tdp-dots{display:flex;gap:4px;align-items:center;padding:8px 0;}",
    ".tdp-dot{width:5px;height:5px;border-radius:50%;background:" + C.text3 + ";animation:tdpBlink 1.2s infinite;}",
    ".tdp-dot:nth-child(2){animation-delay:.2s;}",
    ".tdp-dot:nth-child(3){animation-delay:.4s;}",
    "@keyframes tdpBlink{0%,80%,100%{opacity:.3}40%{opacity:1}}",
    ".tdp-err{align-self:flex-start;max-width:88%;background:rgba(192,57,43,.08);border:1px solid rgba(192,57,43,.3);border-radius:8px;padding:9px 13px;font-size:12.5px;color:#e07060;line-height:1.6;}",
    "#tdp-inp-area{flex-shrink:0;padding:9px 13px 11px;background:" + C.surface + ";border-top:1px solid " + C.goldBorder + ";}",
    "#tdp-tok-row{display:flex;align-items:center;gap:8px;margin-bottom:6px;}",
    "#tdp-tok{font-family:" + C.mono + ";font-size:10px;color:" + C.text3 + ";min-width:92px;transition:color .15s;}",
    "#tdp-tbar{flex:1;height:2px;background:rgba(255,255,255,.08);border-radius:1px;overflow:hidden;}",
    "#tdp-tfill{height:100%;width:0%;border-radius:1px;background:" + C.gold + ";transition:width .15s,background .15s;}",
    "#tdp-tcap{font-family:" + C.mono + ";font-size:10px;color:" + C.text3 + ";}",
    "#tdp-inrow{display:flex;gap:7px;align-items:flex-end;}",
    "#tdp-in{flex:1;background:" + C.bg + ";border:1px solid " + C.borderMid + ";border-radius:9px;padding:8px 12px;font-family:" + C.sans + ";font-size:13px;color:" + C.text + ";resize:none;min-height:40px;max-height:110px;line-height:1.5;outline:none;transition:border-color .15s;}",
    "#tdp-in:focus{border-color:" + C.goldBorder + ";}",
    "#tdp-in::placeholder{color:" + C.text3 + ";}",
    "#tdp-send{width:36px;height:36px;border-radius:50%;flex-shrink:0;border:none;background:linear-gradient(135deg," + C.gold + "," + C.goldDim + ");color:" + C.bg + ";cursor:pointer;font-size:13px;display:flex;align-items:center;justify-content:center;transition:opacity .15s,transform .1s;}",
    "#tdp-send:hover:not(:disabled){transform:scale(1.06);}",
    "#tdp-send:disabled{opacity:.3;cursor:default;}",
    "#tdp-hint{margin-top:5px;font-family:" + C.mono + ";font-size:9.5px;color:" + C.text3 + ";line-height:1.5;}",
    "#tdp-hint span{color:" + C.gold + ";}",
    "#tdp-tswidget{margin-top:6px;min-height:60px;display:flex;align-items:flex-start;justify-content:flex-start;overflow:hidden;}",
    "#tdp-tswidget iframe{transform:scale(.85);transform-origin:left top;}",
    "@media (max-width:640px){#tdp-panel{right:12px;bottom:76px;width:calc(100vw - 24px);height:455px;}#tdp-fab{right:12px;bottom:16px;}}"
  ].join("");

  var HTML =
    '<button id="tdp-fab" aria-label="Ask Sphere about The DRUM Protocols">' +
      FAB_LOGO +
    '</button>' +
    '<div id="tdp-panel" role="dialog" aria-label="Ask Sphere">' +
      '<div id="tdp-hdr">' +
        '<div id="tdp-hdr-l">' +
          SPHERE_LOGO_LG +
          '<div>' +
            '<div id="tdp-title">Ask Sphere · The DRUM Protocols</div>' +
            '<div id="tdp-sub">RAG · Research-backed answers</div>' +
          '</div>' +
        '</div>' +
        '<button id="tdp-close" aria-label="Close">&#x2715;</button>' +
      '</div>' +
      '<div id="tdp-tier">' +
        '<span id="tdp-tbadge" class="tdp-tbadge">RESEARCH</span>' +
        '<span id="tdp-tsrc" class="tdp-tsrc"></span>' +
      '</div>' +
      '<div id="tdp-msgs"></div>' +
      '<div id="tdp-inp-area">' +
        '<div id="tdp-tok-row">' +
          '<span id="tdp-tok">0 / 250 tokens</span>' +
          '<div id="tdp-tbar"><div id="tdp-tfill"></div></div>' +
          '<span id="tdp-tcap">250 tok limit</span>' +
        '</div>' +
        '<div id="tdp-inrow">' +
          '<textarea id="tdp-in" rows="2" placeholder="Ask a question about The DRUM Protocols..."></textarea>' +
          '<button id="tdp-send" aria-label="Send">&#9654;</button>' +
        '</div>' +
        '<div id="tdp-hint"><span>Research-backed answers.</span> Ask clearly and specifically for the best result.</div>' +
        '<div id="tdp-tswidget"></div>' +
      '</div>' +
    '</div>';

  var open = false;
  var loading = false;
  var tsWidget = null;
  var currentTSToken = null;

  function el(id) { return document.getElementById(id); }

  function esc(s) {
    return String(s || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function estimateTokens(text) {
    var w = (text || "").trim().split(/\s+/).filter(function (x) { return x.length > 0; });
    return Math.ceil(w.length * 1.3);
  }

  function scrollMsgs() {
    var m = el("tdp-msgs");
    if (m) m.scrollTop = m.scrollHeight;
  }

  function formatCitation(source) {
    if (!source) return "";
    var title = source.title || source.id || "Source";
    var year = source.year ? " (" + source.year + ")" : "";
    return title + year;
  }

  function init() {
    var font = document.createElement("link");
    font.rel = "stylesheet";
    font.href = "https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap";
    document.head.appendChild(font);

    var style = document.createElement("style");
    style.textContent = CSS;
    document.head.appendChild(style);

    var wrap = document.createElement("div");
    wrap.innerHTML = HTML;
    document.body.appendChild(wrap);

    if (TURNSTILE_KEY && !/^YOUR_|^0x000/.test(TURNSTILE_KEY)) {
      var ts = document.createElement("script");
      ts.src = "https://challenges.cloudflare.com/turnstile/v0/api.js";
      ts.async = true;
      ts.defer = true;
      ts.onload = function () {
        if (window.turnstile) {
          tsWidget = window.turnstile.render("#tdp-tswidget", {
            sitekey: TURNSTILE_KEY,
            theme: "dark",
            size: "compact",
            callback: function (t) { currentTSToken = t; },
            "expired-callback": function () { currentTSToken = null; },
            "error-callback": function () { currentTSToken = null; }
          });
        }
      };
      document.head.appendChild(ts);
    }

    bindEvents();
    addWelcome();
  }

  function bindEvents() {
    el("tdp-fab").addEventListener("click", togglePanel);
    el("tdp-close").addEventListener("click", togglePanel);
    el("tdp-send").addEventListener("click", send);

    var inp = el("tdp-in");
    inp.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        send();
      }
    });
    inp.addEventListener("input", function () {
      autoResize(inp);
      updateTokenBar(inp.value);
    });
  }

  function togglePanel() {
    open = !open;
    el("tdp-panel").classList.toggle("tdp-open", open);
    if (open) setTimeout(function () { el("tdp-in").focus(); }, 220);
  }

  function autoResize(inp) {
    inp.style.height = "auto";
    inp.style.height = Math.min(inp.scrollHeight, 110) + "px";
  }

  function updateTokenBar(text) {
    var cap = 250;
    var t = estimateTokens(text);
    var pct = Math.min(100, (t / cap) * 100);
    var te = el("tdp-tok"), fill = el("tdp-tfill");
    te.textContent = t + " / " + cap + " tokens";
    fill.style.width = pct + "%";
    fill.style.background = t > cap ? C.danger : C.gold;
  }

  function addWelcome() {
    var msgs = el("tdp-msgs");
    var div = document.createElement("div");
    div.className = "tdp-ma";
    div.innerHTML =
      '<div class="tdp-mah">' + SPHERE_LOGO_SM + '<span class="tdp-man">SPHERE</span><span class="tdp-mam">research library</span></div>' +
      '<div class="tdp-mab tdp-welcome">' +
      'Ask about how The DRUM Protocols work, how to choose a series or intensity level, what to expect during a session, or the research behind rhythmic regulation.' +
      '</div>';
    msgs.appendChild(div);
  }

  function addUserMsg(text) {
    var div = document.createElement("div");
    div.className = "tdp-mu";
    div.textContent = text;
    el("tdp-msgs").appendChild(div);
    scrollMsgs();
  }

  function addTyping(label) {
    var div = document.createElement("div");
    div.id = "tdp-typing";
    div.className = "tdp-ma";
    div.innerHTML =
      '<div class="tdp-mah">' + SPHERE_LOGO_SM + '<span class="tdp-man">SPHERE</span><span class="tdp-mam">' + (label || "searching research library…") + '</span></div>' +
      '<div class="tdp-mab"><div class="tdp-dots"><div class="tdp-dot"></div><div class="tdp-dot"></div><div class="tdp-dot"></div></div></div>';
    el("tdp-msgs").appendChild(div);
    scrollMsgs();
  }

  function removeTyping() {
    var t = el("tdp-typing");
    if (t) t.remove();
  }

  function showTierBadge(sources) {
    var tierEl = el("tdp-tier");
    var srcEl = el("tdp-tsrc");
    if (!tierEl) return;

    if (sources && sources.length) {
      srcEl.textContent = sources.slice(0, 3).map(formatCitation).join(" · ");
    } else {
      srcEl.textContent = "";
    }

    tierEl.classList.add("tdp-vis");
  }

  function addAssistantMsg(answer, sources, runId) {
    var msgs = el("tdp-msgs");
    var srcHTML = "";

    if (sources && sources.length) {
      srcHTML = '<div class="tdp-src">';
      sources.forEach(function (s) {
        var label = formatCitation(s);
        if (label) srcHTML += '<span class="tdp-srctag">' + esc(label) + '</span>';
      });
      srcHTML += '</div>';
    }

    var div = document.createElement("div");
    div.className = "tdp-ma";
    div.innerHTML =
      '<div class="tdp-mah">' + SPHERE_LOGO_SM + '<span class="tdp-man">SPHERE</span><span class="tdp-mam">research-based answer</span></div>' +
      '<div class="tdp-mab">' + esc(answer) + '</div>' +
      srcHTML +
      '<div class="tdp-fb">' +
        '<button class="tdp-fbb" onclick="window._tdpFb(this,\'up\',\'' + runId + '\')">&#128077;</button>' +
        '<button class="tdp-fbb" onclick="window._tdpFb(this,\'dn\',\'' + runId + '\')">&#128078;</button>' +
      '</div>';
    msgs.appendChild(div);
    scrollMsgs();
  }

  function addError(msg) {
    var div = document.createElement("div");
    div.className = "tdp-err";
    div.textContent = msg;
    el("tdp-msgs").appendChild(div);
    scrollMsgs();
  }

  function send() {
    if (loading) return;

    var inp = el("tdp-in");
    var question = inp.value.trim();
    if (!question) return;

    if (estimateTokens(question) > 250) {
      addError("Your question is over the 250-token limit. Please shorten it slightly.");
      return;
    }

    inp.value = "";
    inp.style.height = "auto";
    updateTokenBar("");
    loading = true;
    el("tdp-send").disabled = true;

    addUserMsg(question);
    addTyping("searching research library…");

    var payload = { question: question };
    if (currentTSToken) payload.turnstileToken = currentTSToken;

    fetch(WORKER_URL.replace(/\/$/, "") + "/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    })
      .then(function (r) {
        return r.json().then(function (d) {
          return { ok: r.ok, data: d };
        });
      })
      .then(function (res) {
        removeTyping();

        if (!res.ok) {
          addError(res.data.message || res.data.error || "Something went wrong. Please try again.");
          return;
        }

        var data = res.data;
        showTierBadge(data.sources || []);
        addAssistantMsg(
          data.answer || "No answer returned.",
          data.sources || [],
          String(Date.now())
        );

        if (window.turnstile && tsWidget !== null) {
          window.turnstile.reset(tsWidget);
          currentTSToken = null;
        }
      })
      .catch(function () {
        removeTyping();
        addError("Connection error. Please try again.");
      })
      .finally(function () {
        loading = false;
        el("tdp-send").disabled = false;
      });
  }

  window._tdpFb = function (btn, direction, runId) {
    var siblings = btn.parentElement.querySelectorAll(".tdp-fbb");
    siblings.forEach(function (b) { b.classList.remove("tdp-voted"); });
    btn.classList.add("tdp-voted");

    fetch(WORKER_URL.replace(/\/$/, "") + "/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        runId: runId,
        signal: direction === "up" ? 1 : -1
      })
    }).catch(function () {});
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
}());