/* ===================================================================
   MLB Win Probability — shared JavaScript utilities
   ===================================================================
   Sortable tables, hamburger menu, model selector, helpers.
   =================================================================== */

/* ── Hamburger menu toggle ─────────────────────────────────────────── */
function toggleMobileNav() {
  const nav = document.querySelector(".site-nav");
  if (nav) nav.classList.toggle("open");
}

document.addEventListener("click", function (e) {
  const nav = document.querySelector(".site-nav");
  const hamburger = document.querySelector(".hamburger");
  if (
    nav &&
    nav.classList.contains("open") &&
    !nav.contains(e.target) &&
    !hamburger.contains(e.target)
  ) {
    nav.classList.remove("open");
  }
});

/* ── Sortable tables ───────────────────────────────────────────────── */

/**
 * Make a <table> element sortable by clicking its headers.
 *
 * Usage: call `makeSortable(tableElement)` after the table is in the DOM.
 * Each <th> that should be sortable must have a `data-sort` attribute whose
 * value is one of: "string", "number", "date", "percent".
 *
 * The sort cycles: first click = ascending, second = descending, third = reset.
 */
function makeSortable(table) {
  if (!table || table.dataset.sortBound) return;
  table.dataset.sortBound = "1";

  const headers = table.querySelectorAll("th[data-sort]");
  headers.forEach(function (th, colIdx) {
    th.addEventListener("click", function () {
      _sortTable(table, th, colIdx);
    });
  });
}

function _sortTable(table, clickedTh, colIdx) {
  const tbody = table.querySelector("tbody");
  if (!tbody) return;

  const headers = table.querySelectorAll("th[data-sort]");
  const currentDir = clickedTh.classList.contains("sort-asc")
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

/**
 * Scan the page for all tables and attach sorting.
 * Safe to call multiple times (idempotent per table).
 */
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

/* ── API helper ────────────────────────────────────────────────────── */
function api(url) {
  return fetch(url).then(function (r) {
    if (!r.ok) throw new Error("API error " + r.status + ": " + r.statusText);
    return r.json();
  });
}

/* ── Timing utilities ──────────────────────────────────────────────── */

/**
 * Fetch a URL and return { data, serverMs, clientMs }.
 *
 * @param {string} url           API endpoint to fetch.
 * @param {string|null} timingId Optional element ID where timing badge is shown.
 * @returns {Promise<*>}         The parsed JSON response data.
 */
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

/**
 * Display timing information in an element.
 *
 * @param {string|HTMLElement} elOrId  Element or element ID.
 * @param {string|null} serverMs      Server processing time from X-Process-Time-Ms header.
 * @param {number|null} clientMs      Client-side total time from performance.now().
 */
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

/* ── Auto-init on DOMContentLoaded ─────────────────────────────────── */
document.addEventListener("DOMContentLoaded", function () {
  initAllSortables();
  initModelSelector();
});
