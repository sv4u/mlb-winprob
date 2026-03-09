/* ===================================================================
   MLB Prediction System — shared JavaScript utilities (v2)
   ===================================================================
   Sortable tables, hamburger menu, theme toggle, model selector,
   timezone helpers, loading states, and accessibility improvements.
   =================================================================== */

/* ── Theme (Dracula dark / Ayu light) — modes: light, dark, system, time, sun ── */
var THEME_MODE_KEY = "mlb-predict-theme-mode";
var SUN_COORDS_KEY = "mlb-predict-sun-coords";
var THEME_TIME_DARK_START = 18;
var THEME_TIME_DARK_END = 6;

function getSystemPrefersDark() {
  if (typeof window === 'undefined' || !window.matchMedia) return false;
  return window.matchMedia('(prefers-color-scheme: dark)').matches;
}

function getTimeBasedTheme() {
  var hour = new Date().getHours();
  if (hour >= THEME_TIME_DARK_END && hour < THEME_TIME_DARK_START) return 'light';
  return 'dark';
}

function getSunriseSunsetTheme(lat, lng) {
  var now = new Date();
  var sunriseHour = getSunriseHourLocal(now, lat, lng);
  var sunsetHour = getSunsetHourLocal(now, lat, lng);
  if (sunriseHour == null || sunsetHour == null) return getTimeBasedTheme();
  var hour = now.getHours() + now.getMinutes() / 60 + now.getSeconds() / 3600;
  if (hour >= sunriseHour && hour < sunsetHour) return 'light';
  return 'dark';
}

function getSunriseHourLocal(date, lat, lng) {
  var rad = Math.PI / 180;
  var n = Math.floor((date - new Date(date.getFullYear(), 0, 0)) / 86400000);
  var decl = 23.45 * Math.sin(rad * (360 / 365) * (n + 284));
  var latRad = lat * rad;
  var declRad = decl * rad;
  var cosOmega = -Math.tan(latRad) * Math.tan(declRad) - 0.0145 / (Math.cos(latRad) * Math.cos(declRad));
  if (cosOmega > 1 || cosOmega < -1) return cosOmega > 0 ? 12 : null;
  var omega = Math.acos(cosOmega) / rad;
  var solarHour = 12 - (2 * omega) / 15 / 2;
  var tzOffsetHours = -date.getTimezoneOffset() / 60;
  var tzMeridian = tzOffsetHours * 15;
  var correction = (lng - tzMeridian) / 15;
  return solarHour + correction;
}

function getSunsetHourLocal(date, lat, lng) {
  var rad = Math.PI / 180;
  var n = Math.floor((date - new Date(date.getFullYear(), 0, 0)) / 86400000);
  var decl = 23.45 * Math.sin(rad * (360 / 365) * (n + 284));
  var latRad = lat * rad;
  var declRad = decl * rad;
  var cosOmega = -Math.tan(latRad) * Math.tan(declRad) - 0.0145 / (Math.cos(latRad) * Math.cos(declRad));
  if (cosOmega > 1 || cosOmega < -1) return cosOmega > 0 ? 12 : null;
  var omega = Math.acos(cosOmega) / rad;
  var solarHour = 12 + (2 * omega) / 15 / 2;
  var tzOffsetHours = -date.getTimezoneOffset() / 60;
  var tzMeridian = tzOffsetHours * 15;
  var correction = (lng - tzMeridian) / 15;
  return solarHour + correction;
}

function getStoredThemeMode() {
  try {
    var s = localStorage.getItem(THEME_MODE_KEY);
    if (s === 'light' || s === 'dark' || s === 'system' || s === 'time' || s === 'sun') return s;
    var legacy =
      localStorage.getItem("mlb-winprob-theme-mode") ||
      localStorage.getItem("mlb-winprob-theme");
    if (legacy === 'light' || legacy === 'dark') return legacy;
  } catch (_) {}
  return 'system';
}

function resolveTheme(mode) {
  if (mode === 'light') return 'light';
  if (mode === 'dark') return 'dark';
  if (mode === 'system') return getSystemPrefersDark() ? 'dark' : 'light';
  if (mode === 'time') return getTimeBasedTheme();
  if (mode === 'sun') {
    try {
      var coords = localStorage.getItem(SUN_COORDS_KEY);
      if (coords) {
        var p = JSON.parse(coords);
        if (typeof p.lat === 'number' && typeof p.lng === 'number') return getSunriseSunsetTheme(p.lat, p.lng);
      }
    } catch (_) {}
    return getTimeBasedTheme();
  }
  return 'dark';
}

function applyResolvedTheme(theme) {
  var v = theme === 'light' ? 'light' : 'dark';
  if (document.documentElement) document.documentElement.setAttribute('data-theme', v);
  updateThemeUI();
}

function updateThemeUI() {
  var theme = document.documentElement.getAttribute('data-theme');
  var mode = getStoredThemeMode();
  var icon = document.getElementById('theme-current-icon');
  var select = document.getElementById('theme-mode-select');
  if (icon) icon.textContent = theme === 'light' ? '☀' : '🌙';
  if (select && select.value !== mode) select.value = mode;
}

function setThemeMode(mode) {
  var v = mode === 'light' || mode === 'dark' || mode === 'system' || mode === 'time' || mode === 'sun' ? mode : 'system';
  try { localStorage.setItem(THEME_MODE_KEY, v); } catch (_) {}
  if (v === 'sun') {
    try {
      var coords = localStorage.getItem(SUN_COORDS_KEY);
      if (coords) {
        applyResolvedTheme(resolveTheme(v));
        startThemeInterval();
        updateThemeUI();
        return;
      }
    } catch (_) {}
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        function (pos) {
          var p = { lat: pos.coords.latitude, lng: pos.coords.longitude };
          try { localStorage.setItem(SUN_COORDS_KEY, JSON.stringify(p)); } catch (_) {}
          applyResolvedTheme(resolveTheme('sun'));
          startThemeInterval();
          updateThemeUI();
        },
        function () {
          applyResolvedTheme(getTimeBasedTheme());
          startThemeInterval();
          updateThemeUI();
        },
        { timeout: 8000, maximumAge: 86400000 }
      );
      return;
    }
    applyResolvedTheme(getTimeBasedTheme());
    startThemeInterval();
  } else {
    applyResolvedTheme(resolveTheme(v));
    if (v === 'system') {
      attachSystemThemeListener();
      clearThemeInterval();
    }
    if (v === 'time' || v === 'sun') startThemeInterval();
    if (v === 'light' || v === 'dark') clearThemeInterval();
    updateThemeUI();
  }
}

var _themeIntervalId = null;
var _systemListenerAttached = false;

function clearThemeInterval() {
  if (_themeIntervalId != null) {
    clearInterval(_themeIntervalId);
    _themeIntervalId = null;
  }
}

function startThemeInterval() {
  clearThemeInterval();
  _themeIntervalId = setInterval(function () {
    var mode = getStoredThemeMode();
    if (mode !== 'time' && mode !== 'sun') return;
    applyResolvedTheme(resolveTheme(mode));
  }, 60000);
}

function attachSystemThemeListener() {
  if (_systemListenerAttached || !window.matchMedia) return;
  _systemListenerAttached = true;
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function () {
    if (getStoredThemeMode() !== 'system') return;
    applyResolvedTheme(resolveTheme('system'));
  });
}

(function initTheme() {
  var mode = getStoredThemeMode();
  var theme = resolveTheme(mode);
  if (document.documentElement) document.documentElement.setAttribute('data-theme', theme);
  if (mode === 'system') attachSystemThemeListener();
  if (mode === 'time' || mode === 'sun') startThemeInterval();
})();

function injectThemeToggle() {
  var header = document.querySelector('.site-header');
  if (!header || document.getElementById('theme-toggle-wrap')) return;
  var wrap = document.createElement('div');
  wrap.id = 'theme-toggle-wrap';
  wrap.className = 'theme-toggle-wrap';
  var icon = document.createElement('span');
  icon.id = 'theme-current-icon';
  icon.className = 'theme-current-icon';
  icon.setAttribute('aria-hidden', 'true');
  icon.textContent = document.documentElement.getAttribute('data-theme') === 'light' ? '☀' : '🌙';
  var select = document.createElement('select');
  select.id = 'theme-mode-select';
  select.className = 'theme-mode-select';
  select.setAttribute('aria-label', 'Theme mode');
  select.innerHTML = '<option value="light">Light (Ayu)</option><option value="dark">Dark (Dracula)</option><option value="system">System</option><option value="time">Time (6pm–6am)</option><option value="sun">Sunrise / sunset</option>';
  select.value = getStoredThemeMode();
  select.addEventListener('change', function () { setThemeMode(select.value); });
  wrap.appendChild(icon);
  wrap.appendChild(select);
  header.insertBefore(wrap, header.querySelector('.site-nav') || header.firstChild);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', injectThemeToggle);
} else {
  injectThemeToggle();
}

/* ── Timezone Detection & Formatting ──────────────────────────────── */

const USER_TZ = Intl.DateTimeFormat().resolvedOptions().timeZone;
const USER_TZ_ABBR = (function() {
  try {
    var parts = new Date().toLocaleTimeString('en-US', { timeZoneName: 'short' }).split(' ');
    return parts[parts.length - 1];
  } catch (_) { return ''; }
})();

/**
 * Format a UTC/ISO datetime string into the user's local timezone.
 * Returns an object with { date, time, full, relative, tzAbbr }.
 */
function formatLocalTime(isoString) {
  if (!isoString) return { date: '—', time: '—', full: '—', relative: '', tzAbbr: '' };
  var d = new Date(isoString);
  if (isNaN(d.getTime())) return { date: '—', time: '—', full: '—', relative: '', tzAbbr: '' };
  var dateStr = d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: USER_TZ });
  var timeStr = d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', timeZone: USER_TZ });
  var fullStr = d.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric', year: 'numeric', timeZone: USER_TZ })
    + ' ' + timeStr;
  var tzAbbr = '';
  try {
    var tParts = d.toLocaleTimeString('en-US', { timeZoneName: 'short', timeZone: USER_TZ }).split(' ');
    tzAbbr = tParts[tParts.length - 1];
  } catch (_) {}
  return { date: dateStr, time: timeStr, full: fullStr, relative: relativeTime(d), tzAbbr: tzAbbr };
}

function relativeTime(date) {
  var now = new Date();
  var diff = date.getTime() - now.getTime();
  var absDiff = Math.abs(diff);
  var mins = Math.floor(absDiff / 60000);
  var hours = Math.floor(absDiff / 3600000);
  var days = Math.floor(absDiff / 86400000);
  if (diff > 0) {
    if (mins < 60) return 'in ' + mins + 'm';
    if (hours < 24) return 'in ' + hours + 'h';
    if (days === 1) return 'tomorrow';
    if (days < 7) return 'in ' + days + 'd';
    return '';
  }
  if (mins < 60) return mins + 'm ago';
  if (hours < 24) return hours + 'h ago';
  if (days === 1) return 'yesterday';
  return '';
}

function formatGameDate(dateStr) {
  if (!dateStr) return '—';
  var d = new Date(dateStr + 'T12:00:00');
  return d.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric', timeZone: USER_TZ });
}

/* ── Footer timezone badge ────────────────────────────────────────── */
function initFooterTimezone() {
  var els = document.querySelectorAll('.footer-tz-value');
  els.forEach(function(el) {
    el.textContent = USER_TZ.replace(/_/g, ' ') + ' (' + USER_TZ_ABBR + ')';
  });
}

/* ── Hamburger menu toggle ─────────────────────────────────────────── */
function toggleMobileNav() {
  var nav = document.querySelector(".site-nav");
  if (nav) nav.classList.toggle("open");
}

document.addEventListener("click", function (e) {
  var nav = document.querySelector(".site-nav");
  var hamburger = document.querySelector(".hamburger");
  if (
    nav &&
    nav.classList.contains("open") &&
    !nav.contains(e.target) &&
    hamburger &&
    !hamburger.contains(e.target)
  ) {
    nav.classList.remove("open");
  }
});

/* ── Sortable tables ───────────────────────────────────────────────── */

function makeSortable(table) {
  if (!table || table.dataset.sortBound) return;
  table.dataset.sortBound = "1";

  var headers = table.querySelectorAll("th[data-sort]");
  headers.forEach(function (th, colIdx) {
    th.addEventListener("click", function () {
      _sortTable(table, th, colIdx);
    });
  });
}

function _sortTable(table, clickedTh, colIdx) {
  var tbody = table.querySelector("tbody");
  if (!tbody) return;

  var headers = table.querySelectorAll("th[data-sort]");
  var currentDir = clickedTh.classList.contains("sort-asc")
    ? "asc"
    : clickedTh.classList.contains("sort-desc")
      ? "desc"
      : "none";

  headers.forEach(function (h) {
    h.classList.remove("sort-asc", "sort-desc");
  });

  var newDir;
  if (currentDir === "none") newDir = "asc";
  else if (currentDir === "asc") newDir = "desc";
  else newDir = "none";

  if (newDir === "none") {
    var rows = Array.from(tbody.querySelectorAll("tr"));
    rows.sort(function (a, b) {
      return (
        (parseInt(a.dataset.origIdx, 10) || 0) -
        (parseInt(b.dataset.origIdx, 10) || 0)
      );
    });
    rows.forEach(function (r) {
      tbody.appendChild(r);
    });
    return;
  }

  clickedTh.classList.add("sort-" + newDir);

  var realCol = _resolveColIndex(table, clickedTh);
  var sortType = clickedTh.dataset.sort || "string";

  var rows = Array.from(tbody.querySelectorAll("tr"));
  if (!rows[0] || rows[0].dataset.origIdx == null) {
    rows.forEach(function (r, i) {
      r.dataset.origIdx = i;
    });
  }

  rows.sort(function (a, b) {
    var cellA = a.cells[realCol];
    var cellB = b.cells[realCol];
    if (!cellA || !cellB) return 0;

    var vA = (cellA.dataset.sortValue || cellA.textContent).trim();
    var vB = (cellB.dataset.sortValue || cellB.textContent).trim();

    var cmp = _compareValues(vA, vB, sortType);
    return newDir === "desc" ? -cmp : cmp;
  });

  rows.forEach(function (r) {
    tbody.appendChild(r);
  });
}

function _resolveColIndex(table, th) {
  var headerRow = th.parentElement;
  var idx = 0;
  for (var i = 0; i < headerRow.children.length; i++) {
    if (headerRow.children[i] === th) return idx;
    idx += parseInt(headerRow.children[i].colSpan || 1, 10);
  }
  return idx;
}

function _compareValues(a, b, type) {
  if (a === b) return 0;
  if (a === "" || a === "—" || a === "-") return 1;
  if (b === "" || b === "—" || b === "-") return -1;

  switch (type) {
    case "number": {
      var na = parseFloat(a.replace(/[^0-9.\-+]/g, ""));
      var nb = parseFloat(b.replace(/[^0-9.\-+]/g, ""));
      if (isNaN(na)) return 1;
      if (isNaN(nb)) return -1;
      return na - nb;
    }
    case "percent": {
      var pa = parseFloat(a.replace("%", ""));
      var pb = parseFloat(b.replace("%", ""));
      if (isNaN(pa)) return 1;
      if (isNaN(pb)) return -1;
      return pa - pb;
    }
    case "date": {
      var da = new Date(a);
      var db = new Date(b);
      return da - db;
    }
    default:
      return a.localeCompare(b, undefined, {
        numeric: true,
        sensitivity: "base",
      });
  }
}

function initAllSortables() {
  document.querySelectorAll("table").forEach(makeSortable);
}

/* ── Model selector ────────────────────────────────────────────────── */

function initModelSelector() {
  var sel = document.getElementById("model-select");
  if (!sel) return;

  fetch("/api/active-model")
    .then(function (r) {
      return r.json();
    })
    .then(function (data) {
      if (data.model_type) sel.value = data.model_type;
    })
    .catch(function () {});

  sel.addEventListener("change", function () {
    var chosen = sel.value;
    var indicator = document.getElementById("model-switching");
    if (indicator) indicator.style.display = "inline";

    fetch("/api/admin/switch-model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_type: chosen }),
    })
      .then(function (r) {
        return r.json();
      })
      .then(function (data) {
        if (indicator) indicator.style.display = "none";
        if (data.ok) {
          location.reload();
        } else {
          alert("Failed to switch model: " + (data.message || "Unknown error"));
        }
      })
      .catch(function (err) {
        if (indicator) indicator.style.display = "none";
        alert("Error switching model: " + err.message);
      });
  });
}

/* ── API helpers ──────────────────────────────────────────────────── */

function api(url) {
  return fetch(url).then(function (r) {
    if (!r.ok) throw new Error("API error " + r.status + ": " + r.statusText);
    return r.json();
  });
}

function timedFetch(url, timingId) {
  var start = performance.now();
  return fetch(url).then(function (r) {
    if (!r.ok) throw new Error("API error " + r.status + ": " + r.statusText);
    var serverMs = r.headers.get("X-Process-Time-Ms");
    return r.json().then(function (data) {
      var clientMs = performance.now() - start;
      if (timingId) {
        showTiming(timingId, serverMs, clientMs);
      }
      return data;
    });
  });
}

function showTiming(elOrId, serverMs, clientMs) {
  var el =
    typeof elOrId === "string" ? document.getElementById(elOrId) : elOrId;
  if (!el) return;
  var parts = [];
  if (clientMs != null) parts.push(Math.round(clientMs) + "ms");
  if (serverMs != null) parts.push("server " + parseFloat(serverMs).toFixed(0) + "ms");
  el.textContent = parts.join(" · ");
  el.classList.add("timing-badge");
  el.style.display = parts.length ? "" : "none";
}

/* ── Probability helpers ──────────────────────────────────────────── */

function probColor(prob) {
  if (prob >= 0.65) return 'var(--green)';
  if (prob <= 0.35) return 'var(--red)';
  return 'var(--accent)';
}

function probConfidence(prob) {
  var dist = Math.abs(prob - 0.5);
  if (dist >= 0.2) return { label: 'Strong', cls: 'text-green' };
  if (dist >= 0.1) return { label: 'Moderate', cls: 'text-accent' };
  if (dist >= 0.05) return { label: 'Lean', cls: 'text-secondary' };
  return { label: 'Toss-up', cls: 'text-muted' };
}

function renderConfidenceMeter(prob, containerEl) {
  var pct = Math.round(prob * 100);
  var conf = probConfidence(prob);
  var color = probColor(prob);
  containerEl.innerHTML =
    '<div class="confidence-meter">' +
    '<div class="confidence-bar"><div class="confidence-fill" style="width:' + pct + '%;background:' + color + '"></div></div>' +
    '<span class="confidence-label" style="color:' + color + '">' + pct + '%</span>' +
    '</div>';
}

/* ── EV formatting helpers ────────────────────────────────────────── */

function fmtOdds(p) { return p > 0 ? '+' + p : '' + p; }

function evBadge(ev) {
  if (ev > 0) return '<span class="ev-chip ev-chip-positive">+EV $' + ev.toFixed(2) + '</span>';
  return '<span class="ev-chip ev-chip-negative">-EV $' + Math.abs(ev).toFixed(2) + '</span>';
}

/* ── Loading state helpers ────────────────────────────────────────── */

function showSkeleton(containerId, count) {
  var el = document.getElementById(containerId);
  if (!el) return;
  var html = '';
  for (var i = 0; i < (count || 3); i++) {
    html += '<div class="skeleton skeleton-card"></div>';
  }
  el.innerHTML = html;
}

function setLoading(el, on) {
  if (typeof el === 'string') el = document.getElementById(el);
  if (!el) return;
  el.style.opacity = on ? '0.4' : '1';
  el.style.pointerEvents = on ? 'none' : '';
}

/* ── Game type badges ─────────────────────────────────────────────── */

function gameTypeBadge(gt) {
  if (gt === 'S') return ' <span class="badge badge-spring" title="Spring Training">Spring</span>';
  return '';
}

/* ── Days until helper ────────────────────────────────────────────── */

function daysUntil(dateStr) {
  var today = new Date();
  today.setHours(0, 0, 0, 0);
  var game = new Date(dateStr + 'T00:00:00');
  return Math.round((game - today) / 86400000);
}

function dateBadge(dateStr) {
  var d = daysUntil(dateStr);
  if (d < 0) return '<span class="badge badge-neutral">Final</span>';
  if (d === 0) return '<span class="badge badge-live">Today</span>';
  if (d === 1) return '<span class="badge badge-spring">Tomorrow</span>';
  if (d <= 7) return '<span class="badge badge-neutral">In ' + d + 'd</span>';
  return '<span style="color:var(--muted);font-size:0.75rem">In ' + d + 'd</span>';
}

/* ── Auto-init on DOMContentLoaded ─────────────────────────────────── */
document.addEventListener("DOMContentLoaded", function () {
  initAllSortables();
  initModelSelector();
  initFooterTimezone();
});
