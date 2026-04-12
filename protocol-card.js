/**
 * protocol-card.js  ·  The Drum Protocols
 * ─────────────────────────────────────────────────────────────────────────────
 * Shared renderer for the per-protocol detail card.
 * Used by both protocol.html (library page) and adviser.html (result screen).
 *
 * MOBILE-FIRST: 80%+ of users are on mobile. Every layout decision starts
 * from a single-column view. Desktop enhancements are additive.
 *
 * Public API
 * ──────────
 *   ProtocolCard.render(opts)
 *
 * opts {
 *   containerId : string           — id of the <div> to render into
 *   code        : string           — protocol code, e.g. 'H-OS-L1'
 *   volCode     : string           — volume code, e.g. '001' (optional)
 *   protocols   : object           — PROTOCOLS map built by the host page
 *   volumes     : object           — VOLUMES map built by the host page
 *   defaultVol  : string           — fallback volume code
 *   stats       : object           — STATS map (optional)
 *   dataDate    : string           — formatted date string (optional)
 *   context     : 'page'|'adviser' — controls footer links (default 'page')
 *   onVolChange : fn(volCode)      — fired when user changes volume (optional)
 * }
 */

const ProtocolCard = (function () {

  /* ─── Inject shared CSS once ─────────────────────────────────────────────── */
  const STYLE_ID = 'dp-card-styles';

  function injectStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const s = document.createElement('style');
    s.id = STYLE_ID;
    s.textContent = `

/* ── Card shell ─────────────────────────────────────────────────────────── */
.dp-card {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

/* ── Protocol header ────────────────────────────────────────────────────── */
.dp-card-hdr {
  padding: 24px 18px 20px;
  position: relative;
  overflow: hidden;
  border-bottom: 1px solid var(--color-border);
}
/* Decorative rings */
.dp-card-hdr::before {
  content: ''; position: absolute; right: -50px; top: -50px;
  width: 220px; height: 220px; border-radius: 50%;
  border: 1px solid rgba(184,134,11,.12); pointer-events: none;
}
.dp-card-hdr::after {
  content: ''; position: absolute; right: -10px; top: -10px;
  width: 130px; height: 130px; border-radius: 50%;
  border: 1px solid rgba(184,134,11,.08); pointer-events: none;
}
.dp-card-hdr.healing     { background: linear-gradient(135deg,#1a4a35 0%,#132e22 100%); }
.dp-card-hdr.thriving    { background: linear-gradient(135deg,#0f3a5c 0%,#09243a 100%); }
.dp-card-hdr.transforming{ background: linear-gradient(135deg,#3d1a3c 0%,#261027 100%); }
.dp-card-hdr-inner { position: relative; z-index: 1; }

/* Adviser-rec pill */
.dp-rec-bar { display: none; margin-bottom: 14px; }
.dp-rec-bar.visible { display: flex; }
.dp-rec-pill {
  display: inline-flex; align-items: center; gap: 7px;
  background: rgba(184,134,11,.18);
  border: 1.5px solid rgba(184,134,11,.70);
  border-radius: 999px;
  padding: 6px 14px 6px 10px;
  font-size: 10px; letter-spacing: .12em; text-transform: uppercase;
  color: var(--dp-gold-light); font-weight: 500;
  backdrop-filter: blur(4px);
  box-shadow: 0 0 14px rgba(184,134,11,.22);
}
.dp-rec-pill::before {
  content: ''; display: block; width: 8px; height: 8px; border-radius: 50%;
  background: var(--dp-gold-light);
  box-shadow: 0 0 6px rgba(212,160,23,.9);
}

.dp-hdr-series {
  font-size: 10px; letter-spacing: .18em; text-transform: uppercase; margin-bottom: 8px;
}
.dp-hdr-series.healing     { color: var(--dp-healing-accent); }
.dp-hdr-series.thriving    { color: var(--dp-thriving-accent); }
.dp-hdr-series.transforming{ color: var(--dp-transforming-accent); }

.dp-hdr-name {
  font-family: 'Cormorant Garamond', serif;
  font-size: clamp(28px, 7vw, 48px);
  font-weight: 600; color: #fff; line-height: 1.05; margin-bottom: 6px;
}
.dp-hdr-meta { font-size: 12px; color: rgba(255,255,255,.5); letter-spacing: .04em; }

/* ── Section card ───────────────────────────────────────────────────────── */
.dp-sec {
  background: var(--dp-slate-dark);
  border: .5px solid var(--color-border-mid);
  border-radius: var(--radius-lg);
  overflow: hidden;
}
.dp-sec-head { padding: 16px 18px 0; }
.dp-sec-lbl {
  font-size: 10px; letter-spacing: .12em; text-transform: uppercase;
  color: var(--color-text-tertiary); margin-bottom: 12px;
}

/* ── Volume selector + video ────────────────────────────────────────────── */
.dp-vol-wrap { padding: 8px 10px; background: rgba(0,0,0,.4); }
.dp-vol-wrap select {
  width: 100%; padding: 10px 14px;
  background: rgba(255,255,255,.06); color: #fff;
  border: .5px solid rgba(255,255,255,.18); border-radius: var(--radius-sm);
  font-size: 15px; font-family: inherit; cursor: pointer; appearance: none;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'%3E%3Cpath d='M1 1l5 5 5-5' stroke='rgba(255,255,255,0.4)' stroke-width='1.5' fill='none' stroke-linecap='round'/%3E%3C/svg%3E");
  background-repeat: no-repeat; background-position: right 14px center; padding-right: 36px;
}
.dp-vol-wrap select:focus { outline: none; border-color: var(--dp-gold-light); }
.dp-vol-wrap select option { background: var(--dp-slate-deeper); color: #fff; }

.dp-video-shell { position: relative; overflow: hidden; background: #000; }
.dp-video-embed { width: 100%; aspect-ratio: 16/9; border: none; display: block; }
.dp-video-ph {
  width: 100%; aspect-ratio: 16/9;
  background: var(--dp-slate-dark);
  display: flex; flex-direction: column;
  align-items: center; justify-content: center; gap: 10px;
}
.dp-vp-icon {
  width: 56px; height: 56px; border-radius: 50%;
  background: rgba(255,255,255,.08);
  display: flex; align-items: center; justify-content: center;
  font-size: 22px; color: rgba(255,255,255,.4);
}
.dp-vp-lbl { font-size: 12px; color: rgba(255,255,255,.3); letter-spacing: .06em; }

/* Mobile advisory below video */
.dp-mob-advisory {
  display: none;
  background: rgba(184,134,11,.08);
  border-top: 2px solid var(--dp-gold-light);
  padding: 11px 14px;
  align-items: flex-start; gap: 10px;
}
.dp-mob-advisory-icon { font-size: 16px; flex-shrink: 0; padding-top: 1px; }
.dp-mob-advisory-text {
  font-size: 12px; line-height: 1.55; color: var(--color-text-secondary);
}
.dp-mob-advisory-text strong { color: var(--dp-gold-light); font-weight: 500; }
@media (max-width: 768px) { .dp-mob-advisory { display: flex; } }

.dp-yt-ext-row {
  display: none;
  font-size: 12px; color: var(--color-text-tertiary);
  text-align: center; padding: 8px 0 2px;
}
.dp-yt-ext-row a { color: var(--dp-gold-light); border-bottom: .5px solid rgba(212,160,23,.3); }

/* ── Survey ─────────────────────────────────────────────────────────────── */
.dp-survey-wrap { padding: 16px 18px 18px; }
.dp-btn-survey {
  display: flex; flex-direction: column;
  padding: 16px 18px;
  border-radius: var(--radius-md);
  border: .5px solid var(--color-border-mid);
  background: var(--dp-slate-deeper);
  text-decoration: none; color: inherit; width: 100%;
  min-height: 72px;
  transition: border-color .15s, background .15s;
  cursor: pointer; font-family: inherit;
}
.dp-btn-survey:hover, .dp-btn-survey:active {
  border-color: var(--color-border-gold); background: var(--dp-slate-mid);
}
.dp-sq-title { font-size: 14px; font-weight: 500; color: var(--color-text-primary); margin-bottom: 3px; }
.dp-sq-sub   { font-size: 12px; color: var(--color-text-secondary); }
.dp-sq-cta   { font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: var(--dp-gold-light); margin-top: 10px; }
.dp-survey-note { font-size: 11px; color: var(--color-text-tertiary); line-height: 1.6; margin-top: 10px; }
.dp-safety-note {
  font-size: 11px; color: var(--color-text-tertiary);
  border-top: .5px solid var(--color-border); padding: 10px 18px;
}
.dp-safety-note a { color: var(--color-text-tertiary); border-bottom: .5px solid rgba(255,255,255,.12); }

/* ── Curve + About ──────────────────────────────────────────────────────── */
/* Mobile: stacked. ≥540px: side-by-side */
.dp-curve-about { display: flex; flex-direction: column; }
@media (min-width: 540px) {
  .dp-curve-about { flex-direction: row; }
  .dp-curve-col  { flex: 0 0 52%; border-right: .5px solid var(--color-border); border-bottom: none !important; }
  .dp-about-col  { flex: 1; }
}
.dp-curve-col {
  padding: 16px 14px 14px 18px;
  border-bottom: .5px solid var(--color-border);
}
.dp-about-col {
  padding: 16px 18px;
  display: flex; flex-direction: column; justify-content: center;
}
.dp-desc-text { font-size: 14px; color: var(--color-text-secondary); line-height: 1.78; }
#dp-curve-svg { width: 100%; display: block; }

/* ── Stats ──────────────────────────────────────────────────────────────── */
.dp-stats-wrap { padding: 0 18px 18px; }
.dp-stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 14px; }
.dp-stat-card {
  background: rgba(255,255,255,.03);
  border: .5px solid var(--color-border);
  border-radius: var(--radius-md); padding: 12px; text-align: center;
}
.dp-stat-val {
  font-family: 'Cormorant Garamond', serif;
  font-size: clamp(22px, 6vw, 30px);
  font-weight: 600; line-height: 1; color: #fff;
}
.dp-stat-lbl {
  font-size: 9px; color: var(--color-text-tertiary);
  letter-spacing: .07em; text-transform: uppercase; margin-top: 3px;
}
.dp-dist-lbl { font-size: 11px; color: var(--color-text-secondary); margin-bottom: 6px; }
.dp-dist-bars { display: flex; align-items: flex-end; gap: 2px; height: 48px; }
.dp-dist-col  { flex: 1; display: flex; flex-direction: column; align-items: center; gap: 2px; }
.dp-dist-bar  { width: 100%; border-radius: 2px 2px 0 0; background: rgba(255,255,255,.12); min-height: 2px; }
.dp-dist-bar.hi { background: var(--dp-gold-light); opacity: .9; }
.dp-dist-num  { font-size: 8px; color: var(--color-text-tertiary); }
.dp-early-badge {
  display: none; font-size: 10px; padding: 3px 10px;
  background: rgba(184,134,11,.08); border: .5px solid rgba(184,134,11,.25);
  border-radius: 20px; color: var(--dp-gold-light); margin-top: 8px; letter-spacing: .04em;
}

/* ── Commentary ─────────────────────────────────────────────────────────── */
.dp-commentary-wrap {
  background: rgba(184,134,11,.04);
  border-top: .5px solid rgba(184,134,11,.15);
  padding: 18px;
}
.dp-commentary-hdr {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 12px; gap: 10px;
}
.dp-commentary-hdr-right { display: flex; flex-direction: column; gap: 3px; align-items: flex-end; }
.dp-commentary-lbl  { font-size: 10px; letter-spacing: .12em; text-transform: uppercase; color: var(--dp-gold-light); }
.dp-commentary-date { font-size: 10px; color: var(--color-text-tertiary); }
.dp-sphere-icon {
  width: 40px; height: 40px; flex-shrink: 0; border-radius: 50%; overflow: hidden;
  border: 1px solid rgba(184,134,11,.22);
  box-shadow: 0 0 12px rgba(184,134,11,.12), 0 3px 8px rgba(0,0,0,.3);
}
.dp-sphere-icon img { width: 100%; height: 100%; object-fit: cover; display: block; }
.dp-commentary-tabs {
  display: flex; margin-bottom: 12px;
  border-bottom: .5px solid rgba(255,255,255,.08);
}
.dp-commentary-tab {
  font-size: 11px; letter-spacing: .06em; text-transform: uppercase;
  padding: 7px 14px 9px; cursor: pointer;
  color: var(--color-text-tertiary);
  border-bottom: 2px solid transparent; margin-bottom: -1px;
  transition: color .15s, border-color .15s;
  background: none; border-top: none; border-left: none; border-right: none;
  font-family: inherit;
  min-height: 40px; /* comfortable tap target */
}
.dp-commentary-tab:hover  { color: var(--color-text-secondary); }
.dp-commentary-tab.active { color: var(--dp-gold-light); border-bottom-color: var(--dp-gold-light); }
.dp-commentary-text {
  font-size: 13px; line-height: 1.78; color: var(--color-text-secondary);
  border-left: 2px solid rgba(184,134,11,.3); padding-left: 12px;
}
.dp-commentary-text.empty { font-style: italic; color: var(--color-text-tertiary); }

/* ── Related protocols ──────────────────────────────────────────────────── */
.dp-related-section { display: none; margin-top: 4px; }
.dp-related-lbl {
  font-size: 10px; letter-spacing: .14em; text-transform: uppercase;
  color: var(--color-text-tertiary); margin-bottom: 8px;
}
.dp-related-grid { display: flex; flex-direction: column; gap: 6px; }
.dp-related-card {
  display: flex; align-items: center; gap: 12px;
  padding: 14px 16px; min-height: 56px;
  border-radius: var(--radius-md);
  border: .5px solid var(--color-border);
  background: var(--dp-slate-deeper);
  text-decoration: none; color: inherit;
  transition: border-color .15s, background .15s;
}
.dp-related-card:hover, .dp-related-card:active {
  border-color: var(--color-border-gold); background: var(--dp-slate-mid);
}
.dp-rel-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.dp-rel-dot.healing     { background: var(--dp-healing-accent); }
.dp-rel-dot.thriving    { background: var(--dp-thriving-accent); }
.dp-rel-dot.transforming{ background: var(--dp-transforming-accent); }
.dp-rel-name { font-size: 13px; color: var(--color-text-primary); }
.dp-rel-meta { font-size: 11px; color: var(--color-text-tertiary); margin-top: 2px; }

/* ── Footer actions ─────────────────────────────────────────────────────── */
.dp-footer-actions { display: flex; flex-direction: column; gap: 8px; margin-top: 4px; }
.dp-btn-secondary {
  display: flex; align-items: center; justify-content: center;
  padding: 15px 20px; min-height: 50px;
  border-radius: var(--radius-md);
  border: .5px solid var(--color-border-mid);
  background: var(--dp-slate-deeper);
  font-size: 11px; letter-spacing: .06em; text-transform: uppercase;
  color: var(--color-text-secondary); text-decoration: none; text-align: center;
  transition: border-color .15s, background .15s, color .15s;
  font-family: inherit; cursor: pointer;
}
.dp-btn-secondary:hover, .dp-btn-secondary:active {
  border-color: var(--color-border-gold);
  color: var(--color-text-primary);
  background: var(--dp-slate-mid);
}

    `;
    document.head.appendChild(s);
  }

  /* ─── SVG curve ──────────────────────────────────────────────────────────── */
  function drawCurve(p, svgEl) {
    if (!svgEl) return;
    const W = 260, H = 148;
    const HZ_MIN = 3, HZ_MAX = 16;
    const pad = { l: 32, r: 12, t: 18, b: 28 };
    const cW = W - pad.l - pad.r, cH = H - pad.t - pad.b;
    const { start, end, dur, move } = p;
    const X_MAX = 95;
    const toX = t => (t / X_MAX) * cW + pad.l;
    const toY = f => pad.t + cH - (f - HZ_MIN) / (HZ_MAX - HZ_MIN) * cH;
    const baseline = pad.t + cH;

    let pd = '', fd = '';
    for (let i = 0; i <= 80; i++) {
      const t = i / 80; let f;
      if (move === 'HOLD') f = start;
      else {
        const sig = 1 / (1 + Math.exp(-10 * (t - 0.382)));
        f = move === 'DESCENT' ? start - (start - end) * sig : start + (end - start) * sig;
      }
      const x = toX(t * dur), y = toY(f);
      pd += i === 0 ? `M${x},${y}` : `L${x},${y}`;
      if (i === 0) fd = `M${x},${baseline} L${x},${y}`; else fd += `L${x},${y}`;
      if (i === 80) fd += ` L${x},${baseline} Z`;
    }

    const q   = id => svgEl.querySelector('#' + id);
    const set = (id, attrs) => {
      const el = q(id); if (!el) return;
      Object.entries(attrs).forEach(([k, v]) =>
        k === 'text' ? (el.textContent = v) : el.setAttribute(k, v)
      );
    };

    set('dp-c-line',    { d: pd });
    set('dp-c-fill',    { d: fd });
    const ix = toX(dur * 0.382);
    set('dp-c-inflect', { x1: ix, x2: ix, y1: pad.t, y2: baseline });
    set('dp-c-yaxis',   { x1: pad.l, x2: pad.l, y1: pad.t, y2: baseline });
    set('dp-c-ytop',    { x: pad.l - 4, y: pad.t + 4,  text: '16Hz' });
    set('dp-c-ybot',    { x: pad.l - 4, y: toY(4) + 4, text: '4Hz'  });
    set('dp-c-lstart',  { x: toX(0) + 4,   y: toY(start) - 5, 'text-anchor': 'start', text: `${start}Hz` });
    set('dp-c-lend',    { x: toX(dur) - 4, y: toY(end) - 5,   'text-anchor': 'end',   text: `${end}Hz`   });
    set('dp-c-lmove',   { x: pad.l + cW / 2, y: pad.t + 10, text: move });

    [{ id:'20', t:20 }, { id:'30', t:30 }, { id:'45', t:45 }, { id:'60', t:60 }, { id:'90', t:90, lbl:'90+' }]
      .forEach(m => {
        const lEl = q('dp-c-m'  + m.id);
        const tEl = q('dp-c-ml' + m.id);
        if (!lEl || !tEl) return;
        const mx  = toX(m.t);
        const own = (m.t === dur) || (m.id === '90' && dur > 60);
        lEl.setAttribute('x1', mx); lEl.setAttribute('x2', mx);
        lEl.setAttribute('y1', pad.t); lEl.setAttribute('y2', baseline);
        lEl.setAttribute('stroke', own ? 'rgba(255,255,255,.18)' : 'rgba(255,255,255,.06)');
        tEl.setAttribute('x', mx); tEl.setAttribute('y', baseline + 11);
        tEl.setAttribute('fill', own ? '#607080' : '#2a3540');
        tEl.setAttribute('font-weight', own ? '500' : 'normal');
        tEl.textContent = m.lbl || m.id;
      });
  }

  /* ─── Stats ──────────────────────────────────────────────────────────────── */
  function renderStats(code, stats, dataDate, root) {
    const s     = stats[code];
    const $     = sel => root.querySelector(sel);
    const badge = $('.dp-early-badge');

    const setVal = (attr, v) => { const el = $(`[data-stat="${attr}"]`); if (el) el.textContent = v; };

    if (!s || s.n === 0) {
      setVal('n', '—'); setVal('avg', '—'); setVal('pos', '—');
      if (badge) { badge.style.display = 'inline-block'; badge.textContent = 'No data yet · be the first to respond'; }
      const bars = $('.dp-dist-bars');
      if (bars) bars.innerHTML = [0,1,2,3,4,5,6,7,8,9,10].map(i =>
        `<div class="dp-dist-col"><div class="dp-dist-bar${i>=7?' hi':''}" style="height:2px"></div><div class="dp-dist-num">${i}</div></div>`
      ).join('');
      return;
    }

    setVal('n',   s.n);
    setVal('avg', s.mean ? s.mean.toFixed(1) : '—');
    setVal('pos', s.pct  ? s.pct + '%'       : '—');

    const bars = $('.dp-dist-bars');
    if (bars) {
      const maxD = Math.max(...s.dist, 1);
      bars.innerHTML = s.dist.map((v, i) => {
        const h = Math.round((v / maxD) * 44) + 2;
        return `<div class="dp-dist-col"><div class="dp-dist-bar${i>=7?' hi':''}" style="height:${h}px"></div><div class="dp-dist-num">${i}</div></div>`;
      }).join('');
    }

    if (badge) {
      badge.style.display = s.n < 20 ? 'inline-block' : 'none';
      if (s.n < 20) badge.textContent = 'Early data · results will deepen with more responses';
    }

    const noData = 'Commentary generates when sufficient listener data is available for this protocol.';
    const plain  = $('.dp-commentary-plain');
    const tech   = $('.dp-commentary-technical');
    if (plain) { plain.textContent = s.spherePlain || noData; plain.classList.toggle('empty', !s.spherePlain); }
    if (tech)  { tech.textContent  = s.sphere      || noData; tech.classList.toggle('empty',  !s.sphere); }

    const dateEl = $('.dp-commentary-date');
    if (dateEl && dataDate) dateEl.textContent = `Analysis ${dataDate}`;
  }

  /* ─── Commentary tabs ────────────────────────────────────────────────────── */
  function switchTab(tab, root) {
    root.querySelector('.dp-commentary-tab[data-tab="plain"]').classList.toggle('active', tab === 'plain');
    root.querySelector('.dp-commentary-tab[data-tab="technical"]').classList.toggle('active', tab === 'technical');
    const plain = root.querySelector('.dp-commentary-plain');
    const tech  = root.querySelector('.dp-commentary-technical');
    if (plain) plain.style.display = tab === 'plain'     ? '' : 'none';
    if (tech)  tech.style.display  = tab === 'technical' ? '' : 'none';
    try { localStorage.setItem('dp-commentary-tab', tab); } catch(e) {}
  }

  /* ─── Video / volume ─────────────────────────────────────────────────────── */
  function renderVideo(p, volCode, volumes, root, onVolChange) {
    const volKeys = Object.keys(volumes);
    const vol     = volumes[volCode] || volumes[volKeys[0]];

    const dropdownHTML = volKeys.length > 1
      ? `<div class="dp-vol-wrap"><select data-dp-vol>
           ${volKeys.map(k =>
             `<option value="${k}"${k === volCode ? ' selected' : ''}>${volumes[k].label} — ${volumes[k].groove}</option>`
           ).join('')}
         </select></div>`
      : '';

    const ytUrl = typeof p.yt === 'object' ? (p.yt[volCode] || '#') : (p.yt || '#');
    const shell   = root.querySelector('.dp-video-shell');
    const extRow  = root.querySelector('.dp-yt-ext-row');
    const extLink = root.querySelector('.dp-yt-ext-link');
    const ytId    = ytUrl !== '#' ? (ytUrl.match(/(?:v=|youtu\.be\/)([^&?/]+)/) || [])[1] : null;

    if (ytId) {
      // Mobile tap-overlay opens YouTube app so audio plays with screen locked
      const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
      const overlay  = isMobile
        ? `<div onclick="window.open('${ytUrl}','_blank')" style="position:absolute;inset:0;z-index:10;cursor:pointer;background:transparent;" title="Open in YouTube"></div>`
        : '';
      shell.innerHTML = `${dropdownHTML}<div style="position:relative">${overlay}<iframe class="dp-video-embed" src="https://www.youtube.com/embed/${ytId}?rel=0" allow="accelerometer;clipboard-write;encrypted-media;gyroscope;picture-in-picture" allowfullscreen></iframe></div>`;
      if (extRow)  extRow.style.display = '';
      if (extLink) extLink.href = ytUrl;
    } else {
      shell.innerHTML = `${dropdownHTML}<div class="dp-video-ph"><div class="dp-vp-icon">▶</div><div class="dp-vp-lbl">${vol ? vol.label : 'Volume'} — coming soon</div></div>`;
      if (extRow) extRow.style.display = 'none';
    }

    // Wire up volume dropdown change
    const sel = shell.querySelector('[data-dp-vol]');
    if (sel) {
      sel.addEventListener('change', function () {
        const newVol = this.value;
        renderVideo(p, newVol, volumes, root, onVolChange);
        const surveyData = p.surveys && p.surveys[newVol];
        const btn = root.querySelector('.dp-survey-link');
        if (btn && surveyData) btn.href = surveyData.survey1s || '#';
        if (onVolChange) onVolChange(newVol);
      });
    }
  }

  /* ─── Related ────────────────────────────────────────────────────────────── */
  function renderRelated(code, protocols, defaultVol, root) {
    const p = protocols[code];
    if (!p) return;
    const groups = {};
    Object.entries(protocols).forEach(([c, pr]) => {
      const key = pr.series + '|' + pr.entry;
      (groups[key] = groups[key] || []).push(c);
    });
    const siblings = (groups[p.series + '|' + p.entry] || []).filter(c => c !== code);
    const section  = root.querySelector('.dp-related-section');
    const grid     = root.querySelector('.dp-related-grid');
    if (!siblings.length || !section || !grid) return;
    section.style.display = 'block';
    grid.innerHTML = siblings.map(rc => {
      const r = protocols[rc];
      if (!r) return '';
      return `<a class="dp-related-card" href="protocol.html?id=${rc}&vol=VOL${defaultVol}">
        <div class="dp-rel-dot ${r.series}"></div>
        <div>
          <div class="dp-rel-name">${r.name}</div>
          <div class="dp-rel-meta">${r.meta}</div>
        </div>
      </a>`;
    }).join('');
  }

  /* ─── HTML skeleton ──────────────────────────────────────────────────────── */
  function buildHTML(code, p, context) {
    const isAdviser = context === 'adviser';
    return `
<div class="dp-card">

  <!-- HEADER -->
  <div class="dp-card-hdr ${p.series}">
    <div class="dp-card-hdr-inner">
      <div class="dp-rec-bar" id="dp-rec-bar">
        <div class="dp-rec-pill">Adviser Recommendation</div>
      </div>
      <div class="dp-hdr-series ${p.series}">${p.series.toUpperCase()} · ${p.entry}</div>
      <div class="dp-hdr-name">${p.name}</div>
      <div class="dp-hdr-meta">${p.meta} · ${code}</div>
    </div>
  </div>

  <!-- VIDEO -->
  <div class="dp-sec" style="overflow:visible">
    <div class="dp-video-shell"></div>
    <div class="dp-mob-advisory">
      <div class="dp-mob-advisory-icon">🎧</div>
      <div class="dp-mob-advisory-text">
        <strong>For uninterrupted listening</strong> — tap
        <strong>Watch on YouTube</strong> in the player above.
        The YouTube app keeps audio running when your screen locks.
      </div>
    </div>
    <div class="dp-yt-ext-row">
      Prefer the YouTube app?
      <a class="dp-yt-ext-link" href="#" target="_blank" rel="noopener">Open on YouTube →</a>
    </div>
  </div>

  <!-- SURVEY -->
  <div class="dp-sec">
    <div class="dp-sec-head"><div class="dp-sec-lbl">Complete a survey after listening</div></div>
    <div class="dp-survey-wrap">
      <a class="dp-btn-survey dp-survey-link" href="#" target="_blank">
        <div class="dp-sq-title">Take the 1-Second or 1-Minute Survey</div>
        <div class="dp-sq-sub">Your call. Thank you!</div>
        <div class="dp-sq-cta">Take Survey →</div>
      </a>
      <div class="dp-survey-note">
        Your response shapes future protocols.<br>
        Every report matters — including ones that didn't work.
      </div>
    </div>
    <div class="dp-safety-note">
      * Safety, health guidance &amp; privacy policy:
      <a href="safety.html">thedrumprotocols · Safety &amp; Privacy</a>
    </div>
  </div>

  <!-- CURVE + ABOUT (stacked on mobile, side-by-side ≥540px) -->
  <div class="dp-sec dp-curve-about">
    <div class="dp-curve-col">
      <div class="dp-sec-lbl">Entrainment curve</div>
      <svg id="dp-curve-svg" viewBox="0 0 260 148">
        <defs>
          <linearGradient id="dp-cgrad" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%"   stop-color="#B8860B" stop-opacity=".2"/>
            <stop offset="100%" stop-color="#B8860B" stop-opacity=".04"/>
          </linearGradient>
        </defs>
        <path id="dp-c-fill"   fill="url(#dp-cgrad)"/>
        <path id="dp-c-line"   fill="none" stroke="#D4A017" stroke-width="2" stroke-linecap="round"/>
        <line id="dp-c-inflect" stroke="#D4A017" stroke-width=".5" stroke-dasharray="3 3" opacity=".35"/>
        <line id="dp-c-yaxis"  stroke="rgba(255,255,255,.08)" stroke-width="1"/>
        <text id="dp-c-ytop"   font-size="8" fill="#3a4a56" font-family="sans-serif" text-anchor="end"/>
        <text id="dp-c-ybot"   font-size="8" fill="#3a4a56" font-family="sans-serif" text-anchor="end"/>
        <line id="dp-c-m20"  stroke="rgba(255,255,255,.06)" stroke-width="1" stroke-dasharray="2 3"/>
        <line id="dp-c-m30"  stroke="rgba(255,255,255,.06)" stroke-width="1" stroke-dasharray="2 3"/>
        <line id="dp-c-m45"  stroke="rgba(255,255,255,.06)" stroke-width="1" stroke-dasharray="2 3"/>
        <line id="dp-c-m60"  stroke="rgba(255,255,255,.06)" stroke-width="1" stroke-dasharray="2 3"/>
        <line id="dp-c-m90"  stroke="rgba(255,255,255,.06)" stroke-width="1" stroke-dasharray="2 3"/>
        <text id="dp-c-ml20" font-size="8" fill="#2e3a44" font-family="sans-serif" text-anchor="middle"/>
        <text id="dp-c-ml30" font-size="8" fill="#2e3a44" font-family="sans-serif" text-anchor="middle"/>
        <text id="dp-c-ml45" font-size="8" fill="#2e3a44" font-family="sans-serif" text-anchor="middle"/>
        <text id="dp-c-ml60" font-size="8" fill="#2e3a44" font-family="sans-serif" text-anchor="middle"/>
        <text id="dp-c-ml90" font-size="8" fill="#2e3a44" font-family="sans-serif" text-anchor="middle"/>
        <text id="dp-c-lstart" font-size="9" fill="#8a9dac" font-family="sans-serif"/>
        <text id="dp-c-lend"   font-size="9" fill="#8a9dac" font-family="sans-serif"/>
        <text id="dp-c-lmove"  font-size="8" fill="#D4A017" font-family="sans-serif" text-anchor="middle" opacity=".8"/>
        <text font-size="8" fill="#2e3a44" font-family="sans-serif" text-anchor="end" x="248" y="146">minutes</text>
      </svg>
    </div>
    <div class="dp-about-col">
      <div class="dp-sec-lbl">About this protocol</div>
      <div class="dp-desc-text">${p.desc}</div>
    </div>
  </div>

  <!-- STATS -->
  <div class="dp-sec">
    <div class="dp-sec-head"><div class="dp-sec-lbl">Listener data</div></div>
    <div class="dp-stats-wrap">
      <div class="dp-stats-grid">
        <div class="dp-stat-card"><div class="dp-stat-val" data-stat="n">—</div><div class="dp-stat-lbl">Responses</div></div>
        <div class="dp-stat-card"><div class="dp-stat-val" data-stat="avg">—</div><div class="dp-stat-lbl">Avg rating</div></div>
        <div class="dp-stat-card"><div class="dp-stat-val" data-stat="pos">—</div><div class="dp-stat-lbl">Rated ≥ 7</div></div>
      </div>
      <div class="dp-dist-lbl">Rating distribution (0 – 10)</div>
      <div class="dp-dist-bars"></div>
      <div class="dp-early-badge" style="display:none"></div>
    </div>
  </div>

  <!-- COMMENTARY -->
  <div class="dp-sec">
    <div class="dp-commentary-wrap">
      <div class="dp-commentary-hdr">
        <div class="dp-sphere-icon" title="Sphere · Statistical Intelligence">
          <img src="SPHERE_ICON.jpg" alt="Sphere">
        </div>
        <div class="dp-commentary-hdr-right">
          <div class="dp-commentary-lbl">Sphere · statistical commentary</div>
          <div class="dp-commentary-date"></div>
        </div>
      </div>
      <div class="dp-commentary-tabs">
        <button class="dp-commentary-tab active" data-tab="plain">What this means for you</button>
        <button class="dp-commentary-tab" data-tab="technical">Technical details</button>
      </div>
      <div class="dp-commentary-plain dp-commentary-text"></div>
      <div class="dp-commentary-technical dp-commentary-text" style="display:none"></div>
    </div>
  </div>

  <!-- RELATED -->
  <div class="dp-related-section">
    <div class="dp-related-lbl">Also in this series</div>
    <div class="dp-related-grid"></div>
  </div>

  <!-- FOOTER ACTIONS -->
  <div class="dp-footer-actions">
    ${isAdviser
      ? `<a class="dp-btn-secondary" href="protocol.html?id=${code}">View full protocol page →</a>`
      : `<a class="dp-btn-secondary" href="adviser.html">← Find your protocol · Protocol Adviser</a>
         <a class="dp-btn-secondary" href="insights.html">View all listener data · Data Insights</a>`
    }
  </div>

</div>`.trim();
  }

  /* ─── Public render ──────────────────────────────────────────────────────── */
  function render(opts) {
    const {
      containerId,
      code,
      protocols,
      volumes,
      defaultVol,
      volCode    = defaultVol,
      stats      = {},
      dataDate   = '',
      context    = 'page',
      onVolChange,
    } = opts;

    const root = document.getElementById(containerId);
    if (!root) { console.error('ProtocolCard: no element #' + containerId); return; }

    const p = protocols[code];
    if (!p) {
      root.innerHTML = '<p style="color:var(--color-text-tertiary);padding:24px;font-size:14px">Protocol not found.</p>';
      return;
    }

    injectStyles();
    root.innerHTML = buildHTML(code, p, context);

    // Adviser-rec badge
    try {
      const rec = sessionStorage.getItem('dp-adviser-rec');
      root.querySelector('#dp-rec-bar').classList.toggle('visible', rec === code);
    } catch(e) {}

    // Survey link for initial volume
    const surveyData = p.surveys && p.surveys[volCode];
    const surveyBtn  = root.querySelector('.dp-survey-link');
    if (surveyBtn && surveyData) surveyBtn.href = surveyData.survey1s || '#';

    // Draw curve
    drawCurve(p, root.querySelector('#dp-curve-svg'));

    // Video + volume dropdown
    renderVideo(p, volCode, volumes, root, onVolChange);

    // Stats + commentary text
    renderStats(code, stats, dataDate, root);

    // Commentary tab wiring
    root.querySelectorAll('.dp-commentary-tab').forEach(btn =>
      btn.addEventListener('click', () => switchTab(btn.dataset.tab, root))
    );
    try {
      const saved = localStorage.getItem('dp-commentary-tab');
      if (saved) switchTab(saved, root);
    } catch(e) {}

    // Related
    renderRelated(code, protocols, defaultVol, root);
  }

  return { render };

})();
