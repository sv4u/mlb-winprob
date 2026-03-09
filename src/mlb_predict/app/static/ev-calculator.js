/* ===================================================================
   EV Calculator — core math, odds conversion, and real-time UI logic
   ===================================================================
   Shared between the standalone /tools/ev-calculator page and the
   embedded widget on game detail pages.
   =================================================================== */

/**
 * Initialise an EV calculator instance bound to a DOM container.
 *
 * @param {string} prefix    Unique ID prefix for all elements inside the
 *                            calculator (e.g. "ev" for standalone, "ev-game"
 *                            for the embedded widget).
 * @param {object} [opts]    Optional overrides:
 *   - defaultOdds      {string}  Initial odds value (default "-110")
 *   - defaultProb      {string}  Initial win probability (default "50")
 *   - defaultBet       {string}  Initial bet amount (default "10")
 *   - defaultFormat    {string}  "american"|"decimal"|"fractional" (default "american")
 */
function initEVCalculator(prefix, opts) {
  opts = opts || {};

  var el = function (id) {
    return document.getElementById(prefix + "-" + id);
  };

  var oddsInput = el("odds");
  var probInput = el("prob");
  var betInput = el("bet");
  var bankrollInput = el("bankroll");
  var kellySlider = el("kelly-slider");
  var kellyLabel = el("kelly-slider-label");

  if (!oddsInput || !probInput || !betInput) return;

  var currentFormat = opts.defaultFormat || "american";

  // ── Format toggle buttons ───────────────────────────────────────
  var formatBtns = document.querySelectorAll(
    '[data-ev-format][data-ev-prefix="' + prefix + '"]',
  );
  formatBtns.forEach(function (btn) {
    btn.addEventListener("click", function () {
      var newFormat = btn.dataset.evFormat;
      if (newFormat === currentFormat) return;

      var currentOdds = parseOdds(oddsInput.value, currentFormat);
      currentFormat = newFormat;

      formatBtns.forEach(function (b) {
        b.classList.remove("active");
      });
      btn.classList.add("active");

      if (currentOdds !== null) {
        oddsInput.value = convertDecimalTo(currentOdds, currentFormat);
      }
      updatePlaceholder();
      recalc();
    });
  });

  // ── Side toggle buttons (game page only) ────────────────────────
  var sideBtns = document.querySelectorAll(
    '[data-ev-side][data-ev-prefix="' + prefix + '"]',
  );
  sideBtns.forEach(function (btn) {
    btn.addEventListener("click", function () {
      var side = btn.dataset.evSide;
      sideBtns.forEach(function (b) {
        b.classList.remove("active-home", "active-away");
      });
      btn.classList.add(side === "home" ? "active-home" : "active-away");

      if (btn.dataset.evProb) {
        probInput.value = btn.dataset.evProb;
      }
      if (btn.dataset.evOdds) {
        var dec = parseOdds(btn.dataset.evOdds, "american");
        oddsInput.value = dec !== null ? convertDecimalTo(dec, currentFormat) : btn.dataset.evOdds;
      }
      recalc();
    });
  });

  // ── Real-time listeners ─────────────────────────────────────────
  [oddsInput, probInput, betInput].forEach(function (inp) {
    inp.addEventListener("input", recalc);
  });
  if (bankrollInput) bankrollInput.addEventListener("input", recalc);
  if (kellySlider) {
    kellySlider.addEventListener("input", function () {
      if (kellyLabel) kellyLabel.textContent = kellySlider.value + "%";
      recalc();
    });
  }

  function updatePlaceholder() {
    var placeholders = {
      american: "-110",
      decimal: "1.91",
      fractional: "10/11",
    };
    oddsInput.placeholder = placeholders[currentFormat] || "-110";
  }

  // ── Recalculate and render ──────────────────────────────────────
  function recalc() {
    var decimalOdds = parseOdds(oddsInput.value, currentFormat);
    var winProb = parseFloat(probInput.value) / 100;
    var betAmt = parseFloat(betInput.value);
    var bankroll = bankrollInput ? parseFloat(bankrollInput.value) : NaN;
    var kellyFrac = kellySlider ? parseInt(kellySlider.value, 10) / 100 : 1;

    var evValueEl = el("ev-value");
    var evVerdictEl = el("ev-verdict");

    if (
      decimalOdds === null ||
      isNaN(winProb) ||
      isNaN(betAmt) ||
      winProb <= 0 ||
      winProb >= 1 ||
      betAmt <= 0 ||
      decimalOdds <= 1
    ) {
      if (evValueEl) {
        evValueEl.textContent = "—";
        evValueEl.className = "ev-primary-value";
      }
      if (evVerdictEl) {
        evVerdictEl.textContent = "";
        evVerdictEl.className = "ev-primary-verdict";
      }
      setMetric("implied-prob", "—");
      setMetric("edge", "—");
      setMetric("roi", "—");
      setMetric("breakeven", "—");
      clearKelly();
      return;
    }

    var results = computeEV(decimalOdds, winProb, betAmt, kellyFrac);

    if (evValueEl) {
      var sign = results.ev >= 0 ? "+" : "";
      evValueEl.textContent = sign + "$" + Math.abs(results.ev).toFixed(2);
      evValueEl.className =
        "ev-primary-value " + (results.ev >= 0 ? "ev-positive" : "ev-negative");
    }
    if (evVerdictEl) {
      evVerdictEl.textContent =
        results.ev >= 0 ? "+EV — Profitable bet" : "−EV — Unprofitable bet";
      evVerdictEl.className =
        "ev-primary-verdict " +
        (results.ev >= 0 ? "ev-verdict-positive" : "ev-verdict-negative");
    }

    setMetric("implied-prob", (results.impliedProb * 100).toFixed(1) + "%");
    var edgeSign = results.edge >= 0 ? "+" : "";
    setMetricColored(
      "edge",
      edgeSign + (results.edge * 100).toFixed(2) + "%",
      results.edge >= 0,
    );
    setMetricColored(
      "roi",
      (results.roi >= 0 ? "+" : "") + results.roi.toFixed(2) + "%",
      results.roi >= 0,
    );
    setMetric("breakeven", (results.breakeven * 100).toFixed(1) + "%");

    renderKelly(results, bankroll);
  }

  function setMetric(name, value) {
    var mEl = el("metric-" + name);
    if (mEl) {
      mEl.textContent = value;
      mEl.className = "ev-metric-value";
    }
  }

  function setMetricColored(name, value, positive) {
    var mEl = el("metric-" + name);
    if (mEl) {
      mEl.textContent = value;
      mEl.className =
        "ev-metric-value " + (positive ? "ev-positive" : "ev-negative");
    }
  }

  function clearKelly() {
    var kellyPctEl = el("kelly-pct");
    var kellyStakePct = el("kelly-stake-pct");
    var kellyWagerEl = el("kelly-wager");
    var kellyNobetEl = el("kelly-nobet");
    if (kellyPctEl) kellyPctEl.textContent = "—";
    if (kellyStakePct) kellyStakePct.textContent = "—";
    if (kellyWagerEl) kellyWagerEl.textContent = "—";
    if (kellyNobetEl) kellyNobetEl.style.display = "none";
  }

  function renderKelly(results, bankroll) {
    var kellyPctEl = el("kelly-pct");
    var kellyStakePct = el("kelly-stake-pct");
    var kellyWagerEl = el("kelly-wager");
    var kellyNobetEl = el("kelly-nobet");
    var kellyStatsEl = el("kelly-stats");

    if (!kellyPctEl) return;

    if (results.kellyPct <= 0) {
      kellyPctEl.textContent = "0%";
      if (kellyStakePct) kellyStakePct.textContent = "0%";
      if (kellyWagerEl) kellyWagerEl.textContent = "$0.00";
      if (kellyNobetEl) {
        kellyNobetEl.style.display = "";
        kellyNobetEl.textContent = "Kelly says no bet — no edge";
      }
      if (kellyStatsEl) kellyStatsEl.style.display = "none";
      return;
    }

    if (kellyNobetEl) kellyNobetEl.style.display = "none";
    if (kellyStatsEl) kellyStatsEl.style.display = "";

    var pctStr = results.kellyPct.toFixed(2) + "%";
    kellyPctEl.textContent = pctStr;
    if (kellyStakePct) kellyStakePct.textContent = pctStr;
    if (kellyWagerEl) {
      if (!isNaN(bankroll) && bankroll > 0) {
        kellyWagerEl.textContent =
          "$" + ((bankroll * results.kellyPct) / 100).toFixed(2);
      } else {
        kellyWagerEl.textContent = "Enter bankroll";
      }
    }
  }

  updatePlaceholder();
  recalc();
}

/* ── Pure math functions ───────────────────────────────────────────── */

/**
 * Parse an odds string in the given format and return decimal odds.
 * Returns null if the input is invalid.
 */
function parseOdds(raw, format) {
  raw = (raw || "").trim();
  if (!raw) return null;

  switch (format) {
    case "american": {
      var n = parseFloat(raw);
      if (isNaN(n) || n === 0) return null;
      return n > 0 ? n / 100 + 1 : 100 / Math.abs(n) + 1;
    }
    case "decimal": {
      var d = parseFloat(raw);
      if (isNaN(d) || d <= 1) return null;
      return d;
    }
    case "fractional": {
      var parts = raw.split("/");
      if (parts.length !== 2) return null;
      var num = parseFloat(parts[0]);
      var den = parseFloat(parts[1]);
      if (isNaN(num) || isNaN(den) || den === 0) return null;
      return num / den + 1;
    }
    default:
      return null;
  }
}

/**
 * Convert decimal odds to a display string in the given format.
 */
function convertDecimalTo(decimalOdds, format) {
  if (decimalOdds === null || decimalOdds <= 1) return "";

  switch (format) {
    case "american": {
      if (decimalOdds >= 2) {
        return "+" + Math.round((decimalOdds - 1) * 100);
      }
      return "" + Math.round(-100 / (decimalOdds - 1));
    }
    case "decimal":
      return decimalOdds.toFixed(2);
    case "fractional": {
      var profit = decimalOdds - 1;
      var best = _bestFraction(profit);
      return best[0] + "/" + best[1];
    }
    default:
      return "";
  }
}

/** Find a clean fractional representation via limited denominator search. */
function _bestFraction(x) {
  var bestNum = Math.round(x);
  var bestDen = 1;
  var bestErr = Math.abs(x - bestNum);
  for (var d = 2; d <= 20; d++) {
    var n = Math.round(x * d);
    var err = Math.abs(x - n / d);
    if (err < bestErr - 0.0001) {
      bestNum = n;
      bestDen = d;
      bestErr = err;
    }
  }
  var g = _gcd(bestNum, bestDen);
  return [bestNum / g, bestDen / g];
}

function _gcd(a, b) {
  a = Math.abs(a);
  b = Math.abs(b);
  while (b) {
    var t = b;
    b = a % b;
    a = t;
  }
  return a;
}

/**
 * Compute all EV-related metrics.
 *
 * @param {number} decimalOdds  Odds in decimal format (e.g. 1.91).
 * @param {number} winProb      Estimated win probability (0–1).
 * @param {number} betAmt       Bet stake in dollars.
 * @param {number} kellyFrac    Kelly fraction multiplier (0–1; 1 = full Kelly).
 * @returns {object} { ev, impliedProb, edge, roi, breakeven, kellyPct }
 */
function computeEV(decimalOdds, winProb, betAmt, kellyFrac) {
  var netProfit = betAmt * (decimalOdds - 1);
  var ev = winProb * netProfit - (1 - winProb) * betAmt;
  var impliedProb = 1 / decimalOdds;
  var edge = winProb - impliedProb;
  var roi = (ev / betAmt) * 100;
  var breakeven = impliedProb;

  // Kelly criterion: f* = (bp - q) / b where b = decimalOdds - 1
  var b = decimalOdds - 1;
  var q = 1 - winProb;
  var fullKelly = (b * winProb - q) / b;
  var kellyPct = Math.max(0, fullKelly * (kellyFrac || 1)) * 100;

  return {
    ev: ev,
    impliedProb: impliedProb,
    edge: edge,
    roi: roi,
    breakeven: breakeven,
    kellyPct: kellyPct,
    fullKellyPct: Math.max(0, fullKelly) * 100,
  };
}
