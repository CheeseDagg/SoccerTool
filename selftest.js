#!/usr/bin/env node
/* selftest.js — headless check of index.html's render layer.
 *
 * WHY THIS FILE EXISTS. The Python side has three selftests and, until today, the
 * daily workflow ran none of them; the browser side had none at all. index.html is
 * ~730 lines that rebuild the scoreline grid, the props board and the verdict table
 * from slate.json, and every one of those was verified only by a human opening the
 * page. Two of the defects fixed today were in exactly that layer:
 *
 *   - the Props tab's empty state said "no fixtures carry player shares yet" whether
 *     the calendar was empty or the player feed was DEAD. It was dead, in all four
 *     leagues, for weeks, and the page said the reassuring thing.
 *   - the Method tab reported how often the MODEL was right when it disagreed with
 *     the closing market (23-31%) and never how often the MARKET was on the same
 *     matches (44.7%). On three outcomes you cannot derive one from the other, so
 *     the table looked mediocre when the underlying result is unambiguous.
 *
 * The harness extracts the SHIPPED <script> block out of index.html rather than
 * testing a copy, stubs the DOM, and renders against both the real committed
 * slate.json and synthetic slates that force each branch.
 */
const fs = require('fs');
const path = require('path');
const HERE = __dirname;

const html = fs.readFileSync(path.join(HERE, 'index.html'), 'utf8');
const blocks = [...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)].map(m => m[1]);
if (blocks.length !== 1) fail(`expected 1 script block, found ${blocks.length}`);
// Strip the boot fetch: the harness drives render() itself.
const src = blocks[0].replace(/fetch\('data\/slate\.json'\)[\s\S]*$/, '');

let failures = 0;
function fail(msg) { console.log('  FAIL: ' + msg); failures++; }
function ok(msg) { console.log('  ok: ' + msg); }
function check(cond, msg) { cond ? ok(msg) : fail(msg); }

/* ---------------------------------------------------------------- DOM stub */
function makeDom() {
  const els = {};
  const mk = id => (els[id] = {
    id, innerHTML: '', textContent: '', style: {}, className: '',
    classList: { add() {}, remove() {}, toggle() {}, contains: () => false },
    appendChild() {}, querySelectorAll: () => [], addEventListener() {},
  });
  const doc = {
    getElementById: id => els[id] || mk(id),
    querySelectorAll: () => [],
    querySelector: () => null,
    createElement: () => mk('_tmp' + Math.random()),
    addEventListener() {},
    body: mk('body'),
  };
  return { els, doc };
}

function runRender(slate) {
  const { els, doc } = makeDom();
  const sandbox = {
    document: doc,
    window: { addEventListener() {}, location: { hash: '' } },
    location: { hash: '' },
    console,
    fetch: () => Promise.reject(new Error('no network in selftest')),
    setTimeout, clearTimeout, Math, JSON, Date, Number, String, Array, Object,
    isFinite, parseFloat, parseInt,
    // render() installs a MutationObserver to re-wrap wide panels. There is no
    // layout here, so it only has to exist and swallow observe().
    MutationObserver: function () { this.observe = function () {}; this.disconnect = function () {}; },
  };
  const names = Object.keys(sandbox);
  const body = src + '\n;render(__SLATE__); return {els:__ELS__};';
  const fn = new Function(...names, '__SLATE__', '__ELS__', body);
  fn(...names.map(n => sandbox[n]), slate, els);
  return els;
}

/* ------------------------------------------------------- 1) the real slate */
console.log('1) render the committed slate.json');
const real = JSON.parse(fs.readFileSync(path.join(HERE, 'data', 'slate.json'), 'utf8'));
let els;
try {
  els = runRender(real);
  ok('render() completed on the real slate without throwing');
} catch (e) {
  fail('render() threw on the real slate: ' + e.message);
}

if (els) {
  const props = els['t-props'] ? els['t-props'].innerHTML : '';
  // This assertion must NOT be "the committed slate has all leagues off". That is
  // true today and the whole point of today's props fix is to make it false, so
  // pinning it would turn this gate red on the first successful run. Assert the
  // page tells the truth about WHATEVER the committed slate says instead.
  const parts = (real.props_src || '').split('·').map(s => s.trim()).filter(Boolean);
  const allOff = parts.length > 0 && parts.every(s => s.includes(':off('));
  console.log('     (committed slate props_src: ' + (real.props_src || '(none)') + ')');
  if (allOff) {
    check(/broken feed, not an empty calendar/.test(props),
      'committed slate has every league off -> the page calls it a DEAD FEED');
  } else {
    check(!/broken feed, not an empty calendar/.test(props),
      'committed slate has a live league -> the page does NOT cry dead feed');
  }
  check(!/no fixtures carry player shares yet/.test(props),
    'the old reassuring empty-state string is gone');
}

/* --------------------------------------- 2) Props empty state, each branch */
console.log('2) Props empty state distinguishes its causes');
const baseSlate = () => JSON.parse(JSON.stringify({
  props_src: '', generated: '2026-08-03T09:30:00Z', slate_date: '2026-08-03',
  leagues: Object.fromEntries(['E0', 'SP1', 'D1', 'F1'].map(k => [k, {
    name: k, n_matches: 100, latest_result: '2026-05-24', home_adv: 0.25,
    rho: -0.05, mu: 0.1, ratings: [], fixtures: [], backtest: null,
  }])),
  note: '', cal: { n: 0 },
}));

function propsHtml(mutate) {
  const s = baseSlate(); mutate(s);
  const e = runRender(s);
  return e['t-props'] ? e['t-props'].innerHTML : '';
}

let h = propsHtml(s => { s.props_src = 'E0:off(A) · SP1:off(B) · D1:off(C) · F1:off(D)'; });
check(/DOWN in all 4 leagues/.test(h) && /broken feed/.test(h),
  'all leagues off  -> "the player-shares source is DOWN in all 4 leagues"');

h = propsHtml(s => { s.props_src = 'E0:120p/20t · SP1:off(B) · D1:110p/18t · F1:pin(90p,14d)'; });
check(/1 of 4 leagues have no player shares/.test(h),
  'one league off   -> names how many of how many, not a blanket message');

h = propsHtml(s => { s.props_src = 'E0:120p/20t · SP1:118p/20t · D1:110p/18t · F1:100p/18t'; });
check(/no fixtures on the board yet/.test(h),
  'feed healthy, no fixtures -> "no fixtures on the board yet"');

h = propsHtml(s => {
  s.props_src = 'E0:120p/20t · SP1:118p/20t · D1:110p/18t · F1:100p/18t';
  s.leagues.E0.fixtures = [{ home: 'Arsenal', away: 'Chelsea', date: '2026-08-08',
    pH: 0.5, pD: 0.25, pA: 0.25, lh: 1.5, la: 1.1, o25: 0.55, btts: 0.5 }];
});
check(/none carry player shares yet/.test(h),
  'feed healthy, fixtures present, no scorers -> the fourth, distinct message');

/* -------------------------------- 3) the disagreement verdict on Method tab */
console.log('3) Method tab reports the market\'s rate on the disagreements');
function methodHtml(bt) {
  const s = baseSlate();
  for (const k of Object.keys(s.leagues)) s.leagues[k].backtest = JSON.parse(JSON.stringify(bt));
  const e = runRender(s);
  return e['t-method'] ? e['t-method'].innerHTML : '';
}

const BT_NEW = { n: 800, acc: 51.0, brier3: 0.59, n_mkt: 800, acc_mkt: 51.0,
  market_acc: 54.5, disagree_n: 100, disagree_model_right: 25.0,
  disagree_market_right: 45.0,
  totals: { n: 800, acc: 57.0, market_acc: 59.0, disagree_n: 150,
            disagree_model_right: 45.0, disagree_market_right: 55.0 } };

h = methodHtml(BT_NEW);
check(/Read this before you take a soccer leg off this board/.test(h),
  'the verdict panel renders when disagree_market_right is present');
check(/right <b>45\.0%<\/b> of the time/.test(h),
  "the market's rate on the disagreements is printed, not left to be inferred");
check(/<b>1\.8x<\/b> better/.test(h),
  'the ratio is computed and stated (45.0 / 25.0 = 1.8x)');
check(/do not sum to 100/.test(h),
  'the panel explains why the two rates are not complements');
check(/>400</.test(h),
  'the disagreement count is summed across the four leagues (4 x 100 = 400)');

// A slate published BEFORE this field existed must not break the page, and must
// not render a panel with "undefined" in it.
const BT_OLD = JSON.parse(JSON.stringify(BT_NEW));
delete BT_OLD.disagree_market_right;
delete BT_OLD.totals.disagree_market_right;
h = methodHtml(BT_OLD);
check(!/Read this before you take a soccer leg/.test(h),
  'an older slate without the field renders no verdict panel (rather than a broken one)');
check(!/undefined|NaN/.test(h),
  'no "undefined"/"NaN" leaks into the Method tab on an older slate');

/* --------------------- 4) the live grader panel on the Calibration tab */
console.log('4) Calibration tab reports the live grader symmetrically');
function calHtml(cal) {
  const s = baseSlate();
  s.cal = cal;
  for (const k of Object.keys(s.leagues)) s.leagues[k].backtest = JSON.parse(JSON.stringify(BT_NEW));
  const e = runRender(s);
  return e['t-cal'] ? e['t-cal'].innerHTML : '';
}

const CAL_NEW = { n: 200, acc: 50.5, brier3: 0.60,
  market: { n: 190, acc: 54.0, model_acc: 50.0, disagree_n: 40,
            disagree_model_right: 25.0, disagree_market_right: 45.0 },
  by_league: { E0: { n: 100, acc: 51.0, brier3: 0.60,
    market: { n: 95, acc: 54.0, model_acc: 50.5, disagree_n: 20,
              disagree_model_right: 25.0, disagree_market_right: 45.0 } } } };

h = calHtml(CAL_NEW);
check(/market right <b>45%<\/b>/.test(h),
  "the live grader prints the MARKET's rate on the disagreements, not only the model's");
check(/model <b>50%<\/b>/.test(h),
  'the live panel quotes the model on the priced subset (model_acc), not over all rows');
check(/carried a closing price/.test(h),
  'the live panel says which match set the two accuracies describe');
check(/do not sum to 100/.test(h),
  'the live panel explains the three-way non-complement too');
check(/\(25% \/ 45%\)/.test(h),
  'the per-league row carries both rates');

// A grader panel written before disagree_market_right existed must still render.
const CAL_OLD = JSON.parse(JSON.stringify(CAL_NEW));
delete CAL_OLD.market.disagree_market_right;
delete CAL_OLD.market.model_acc;
delete CAL_OLD.by_league.E0.market.disagree_market_right;
h = calHtml(CAL_OLD);
check(!/undefined|NaN/.test(h), 'an older cal panel renders with no "undefined"/"NaN"');
check(/model <b>50\.5%<\/b>/.test(h),
  'with model_acc absent the panel falls back to the overall accuracy rather than blank');

/* --------------------------------- 5) the site-wide verdict banner */
console.log('5) verdict banner compares like for like');
// The banner is emitted by verdictHTML() into several tabs (legs, builder, lab,
// calibration) and not into Method, which carries its own panel. Scan every
// rendered element rather than guessing which tab hosts it.
function verdictOf(bt) {
  const s = baseSlate();
  s.cal = CAL_NEW;
  for (const k of Object.keys(s.leagues)) s.leagues[k].backtest = JSON.parse(JSON.stringify(bt));
  const e = runRender(s);
  return Object.values(e).map(x => x.innerHTML || '').join('\n<!--el-->\n');
}
h = verdictOf(BT_NEW);
check(/the market was right <b>45\.0%<\/b>/.test(h),
  "the banner states the market's rate on the disagreements");
check(/same matches, both figures/.test(h),
  'the banner says the two accuracies are over the same match set');
check(/verdict bad/.test(h),
  'the banner flags itself bad because the market beats the model on the disagreements');
// A model that is BETTER on its disagreements must not be flagged bad.
const BT_GOOD = JSON.parse(JSON.stringify(BT_NEW));
BT_GOOD.disagree_model_right = 55.0; BT_GOOD.disagree_market_right = 30.0;
h = verdictOf(BT_GOOD);
check(!/verdict bad/.test(h),
  'a model that wins its disagreements is NOT flagged bad (the test is market>model, not model<50)');

/* ------------------------------------------------ 6) no stale-cache fetches */
console.log('6) the slate cannot be served from browser cache');
// The daily Action republishes data/slate.json at the same URL every morning. A cached
// copy renders perfectly and is simply yesterday's fixtures -- the one failure mode a
// reader has no way to see, on a page whose entire value is that it is today's.
{
  const fetches = [...html.matchAll(/fetch\(\s*'(data\/[^']+)'([^)]*)\)/g)];
  check(fetches.length > 0, 'found the data fetches in index.html');
  const cacheable = fetches.filter(m => !/no-store/.test(m[2])).map(m => m[1]);
  check(cacheable.length === 0,
    "every data fetch passes cache:'no-store'" +
    (cacheable.length ? ' — CACHEABLE: ' + cacheable.join(', ') : ''));
}

/* ------------------------------------------------------------------ report */
console.log(failures ? `\nSOCCER UI SELFTEST: ${failures} FAILURE(S)`
                     : '\nSOCCER UI SELFTEST PASS — props empty-state tells the truth about a '
                       + 'dead feed; the Method tab states the market\'s accuracy on the '
                       + 'disagreements instead of leaving it to be derived');
process.exit(failures ? 1 : 0);
