#!/usr/bin/env node
/**
 * bench-concurrent.js — Fase 1 throughput benchmark.
 *
 * Spawns qwen-serve (or connects to a running one with --no-spawn) and fires
 * bursts of concurrent /v1/systemone requests at several concurrency levels.
 * Reports per-level: throughput (req/s), p50/p95 latency, success rate.
 *
 * Concurrency levels always run with cache-busted states (a nonce appended),
 * so they measure REAL inference throughput. Repeated bodies would silently
 * hit the exact cache and inflate req/s by ~1000x (that was happening before:
 * "peak throughput 888 req/s" while true inference was ~1.2 req/s).
 * The final cache-hit phase repeats one warm body on purpose to measure the
 * cache ceiling.
 *
 * Usage:
 *   node tools/bench-concurrent.js
 *   node tools/bench-concurrent.js --workers 2 --levels 1,2,4,8 --requests 16
 *   node tools/bench-concurrent.js --no-spawn --port 8093
 *   node tools/bench-concurrent.js --binary dist/bin/linux-x64/qwen-serve
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');

const args = process.argv.slice(2);
const val = (name, def) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && args[i + 1] && !args[i + 1].startsWith('--') ? args[i + 1] : def;
};
const flag = (name) => args.includes(`--${name}`);

const modelDir = path.resolve(ROOT, val('model', 'models'));
const port = parseInt(val('port', '8099'), 10);
const workers = parseInt(val('workers', '0'), 10); // 0 = server default
const levels = val('levels', '1,2,4,8').split(',').map((s) => parseInt(s.trim(), 10)).filter((n) => n > 0);
const perLevel = parseInt(val('requests', '16'), 10);
const noSpawn = flag('no-spawn');

let explicitBinary = val('binary', null);
if (!explicitBinary) {
  const defaultBin = process.platform === 'win32'
    ? path.join(ROOT, 'native', 'qwen-serve', 'target', 'release', 'qwen-serve.exe')
    : path.join(ROOT, 'native', 'qwen-serve', 'target', 'release', 'qwen-serve');
  explicitBinary = defaultBin;
}
const binPath = path.resolve(ROOT, explicitBinary);

const STATES = [
  'We were billed twice on the March invoice and want a refund.',
  'The application crashes with a segfault when I open the settings page.',
  'Fui cobrado em duplicidade na minha fatura e quero reembolso.',
  'Me cobraron dos veces en mi factura y quiero un reembolso.',
  'Your service has been down for six hours and nobody answers.',
  'The app freezes and throws an exception on startup.',
  'I was charged the wrong amount on my last invoice.',
  'Quero fazer upgrade do meu plano para o empresarial.',
  'Can you send me a quote for the business tier?',
  'We would like to purchase more seats for our account.',
  'The dashboard shows a 500 error every time I export a report.',
  'Preciso da segunda via do boleto que venceu ontem.',
];

const QUESTIONS = {
  department: {
    type: 'choice',
    instructions: 'Which department should handle this ticket?',
    criteria: {
      billing: 'refunds, charges, payments and invoices',
      tech: 'bugs, crashes, downtime and technical issues',
      sales: 'upgrades, subscriptions, contracts and seat purchases',
    },
  },
};

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function waitForHealth(url, timeoutMs = 120000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(url);
      if (res.ok) {
        const body = await res.json();
        if (body.status === 'ok') return body;
      }
    } catch {}
    await sleep(500);
  }
  throw new Error(`Server failed to respond on ${url} within ${timeoutMs}ms`);
}

function percentile(sorted, p) {
  if (sorted.length === 0) return 0;
  const idx = Math.min(sorted.length - 1, Math.ceil((p / 100) * sorted.length) - 1);
  return sorted[Math.max(0, idx)];
}

let nonce = 0;

async function oneRequest(url, state, unique = true) {
  const t0 = Date.now();
  // Unique states guarantee a cache miss: every request runs real inference.
  const body = { state: unique ? `${state} [bench-${++nonce}]` : state, questions: QUESTIONS };
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const dur = Date.now() - t0;
    if (!res.ok) return { ok: false, dur };
    const data = await res.json();
    const ok = !!data.answers?.department?.choice;
    return { ok, dur };
  } catch {
    return { ok: false, dur: Date.now() - t0 };
  }
}

async function runLevel(url, concurrency, total, states = STATES, unique = true) {
  const latencies = [];
  let ok = 0;
  let idx = 0;
  const t0 = Date.now();

  async function worker() {
    while (true) {
      const i = idx++;
      if (i >= total) return;
      const r = await oneRequest(url, states[i % states.length], unique);
      latencies.push(r.dur);
      if (r.ok) ok++;
    }
  }

  await Promise.all(Array.from({ length: concurrency }, () => worker()));
  const wallMs = Date.now() - t0;
  latencies.sort((a, b) => a - b);
  return {
    concurrency,
    total,
    ok,
    wallMs,
    rps: (ok / (wallMs / 1000)).toFixed(2),
    p50: percentile(latencies, 50).toFixed(0),
    p95: percentile(latencies, 95).toFixed(0),
    max: Math.max(...latencies).toFixed(0),
  };
}

async function main() {
  let child = null;
  const cleanup = () => {
    try {
      if (child && !child.killed) {
        if (process.platform === 'win32') spawn('taskkill', ['/pid', child.pid.toString(), '/f', '/t']);
        else child.kill('SIGTERM');
      }
    } catch {}
  };
  process.on('exit', cleanup);
  process.on('SIGINT', () => { cleanup(); process.exit(1); });

  try {
    if (!noSpawn) {
      if (!fs.existsSync(binPath)) {
        console.error(`[bench] binary not found at: ${binPath}`);
        process.exit(1);
      }
      const srvArgs = ['--model-dir', modelDir, '--port', String(port), '--host', '127.0.0.1'];
      if (workers > 0) srvArgs.push('--workers', String(workers));
      console.log(`[bench] starting: ${binPath} ${srvArgs.join(' ')}`);
      child = spawn(binPath, srvArgs, { stdio: ['ignore', 'inherit', 'inherit'] });
    }

    const healthUrl = `http://127.0.0.1:${port}/health`;
    const health = await waitForHealth(healthUrl);
    console.log(`[bench] server health: ${JSON.stringify(health)}`);

    const apiUrl = `http://127.0.0.1:${port}/v1/systemone`;

    // Warmup: 2 sequential requests (page in weights, fill caches).
    console.log('[bench] warmup (2 sequential)...');
    await oneRequest(apiUrl, STATES[0], false);
    await oneRequest(apiUrl, STATES[1], false);

    console.log(`\n[bench] levels below are cache-busted: every request runs real inference`);
    console.log(`concurrency | ok/total | wall(ms) | req/s | p50(ms) | p95(ms) | max(ms)`);
    console.log(`--------------------------------------------------------------------------------`);
    const rows = [];
    for (const c of levels) {
      const r = await runLevel(apiUrl, c, perLevel);
      rows.push(r);
      console.log(
        `${String(r.concurrency).padStart(11)} | ${String(`${r.ok}/${r.total}`).padStart(8)} | ${String(r.wallMs).padStart(8)} | ${String(r.rps).padStart(5)} | ${String(r.p50).padStart(7)} | ${String(r.p95).padStart(7)} | ${String(r.max).padStart(7)}`
      );
    }
    console.log(`--------------------------------------------------------------------------------`);
    const best = rows.reduce((a, b) => (parseFloat(b.rps) > parseFloat(a.rps) ? b : a), rows[0]);
    console.log(`[bench] peak inference throughput: ${best.rps} req/s at concurrency ${best.concurrency}`);

    // Cache-hit phase: repeat ONE warm body (answered during warmup) at high
    // concurrency. All should be exact-cache hits (~ms, no inference).
    console.log(`\n[bench] cache-hit phase (repeat warm body, expect ~ms)...`);
    const hit = await runLevel(apiUrl, Math.max(...levels), perLevel, [STATES[0]], false);
    console.log(
      `${String(`hit@${hit.concurrency}`).padStart(11)} | ${String(`${hit.ok}/${hit.total}`).padStart(8)} | ${String(hit.wallMs).padStart(8)} | ${String(hit.rps).padStart(5)} | ${String(hit.p50).padStart(7)} | ${String(hit.p95).padStart(7)} | ${String(hit.max).padStart(7)}`
    );
    try {
      const h = await (await fetch(healthUrl)).json();
      console.log(`[bench] cache stats: ${JSON.stringify(h.cache)}`);
    } catch {}

    // Fail loudly if any level lost requests — concurrency must not break correctness.
    const failed = rows.filter((r) => r.ok !== r.total);
    cleanup();
    if (failed.length > 0) {
      console.error(`[bench] FAILED: ${failed.map((r) => `c=${r.concurrency} ok=${r.ok}/${r.total}`).join(', ')}`);
      process.exit(1);
    }
    console.log('[bench] all levels completed with 100% valid responses ✔');
  } catch (err) {
    cleanup();
    console.error('[bench] ERROR:', err);
    process.exit(1);
  }
}

main();
