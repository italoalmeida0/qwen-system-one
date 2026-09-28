#!/usr/bin/env node
/**
 * quick-check.js — verifies that qwen-serve compiles, boots, and answers correctly.
 *
 * Usage:
 *   node tools/quick-check.js
 *   node tools/quick-check.js --binary dist/bin/linux-x64/qwen-serve
 *   node tools/quick-check.js --model models --port 8095
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
  return i >= 0 && args[i + 1] ? args[i + 1] : def;
};

const modelDir = path.resolve(ROOT, val('model', 'models'));
const port = parseInt(val('port', '8093'), 10);
// Optional --ep is forwarded to the spawned server (cpu|auto|dml|coreml|nnapi|qnn);
// --cases N truncates the case list (for slow EP smoke runs on software adapters).
const ep = val('ep', null);
const maxCases = parseInt(val('cases', '0'), 10);

let explicitBinary = val('binary', null);
if (!explicitBinary) {
  const defaultBin = process.platform === 'win32'
    ? path.join(ROOT, 'native', 'qwen-serve', 'target', 'release', 'qwen-serve.exe')
    : path.join(ROOT, 'native', 'qwen-serve', 'target', 'release', 'qwen-serve');
  explicitBinary = defaultBin;
}
const binPath = path.resolve(ROOT, explicitBinary);

if (!fs.existsSync(binPath)) {
  console.error(`[quick-check] binary not found at: ${binPath}`);
  process.exit(1);
}

const ALL_CASES = [
  ['We were billed twice on the March invoice and want a refund.', 'billing'],
  ['The application crashes with a segfault when I open the settings page.', 'tech'],
  ['Fui cobrado em duplicidade na minha fatura e quero reembolso.', 'billing'],
  ['Me cobraron dos veces en mi factura y quiero un reembolso.', 'billing'],
  ['Your service has been down for six hours and nobody answers.', 'tech'],
  ['The app freezes and throws an exception on startup.', 'tech'],
  ['I was charged the wrong amount on my last invoice.', 'billing'],
  ['Quero fazer upgrade do meu plano para o empresarial.', 'sales'],
  ['Can you send me a quote for the business tier?', 'sales'],
  ['We would like to purchase more seats for our account.', 'sales']
];
const CASES = maxCases > 0 ? ALL_CASES.slice(0, maxCases) : ALL_CASES;

const QUESTIONS = {
  department: {
    type: 'choice',
    instructions: 'Which department should handle this ticket?',
    criteria: {
      billing: 'refunds, charges, payments and invoices',
      tech: 'bugs, crashes, downtime and technical issues',
      sales: 'upgrades, subscriptions, contracts and seat purchases'
    }
  }
};

async function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function waitForHealth(url, timeoutMs = 45000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(url);
      if (res.ok) {
        const body = await res.json();
        if (body.status === 'ok') return body;
      }
    } catch {}
    await sleep(250);
  }
  throw new Error(`Server failed to respond on ${url} within ${timeoutMs}ms`);
}

async function main() {
  console.log(`[quick-check] starting qwen-serve...`);
  console.log(`[quick-check] binary: ${binPath}`);
  console.log(`[quick-check] models: ${modelDir}`);
  console.log(`[quick-check] port  : ${port}`);

  const child = spawn(binPath, ['--model-dir', modelDir, '--port', String(port), '--host', '127.0.0.1', ...(ep ? ['--ep', ep] : [])], {
    stdio: ['ignore', 'inherit', 'inherit']
  });

  const cleanup = () => {
    try {
      if (!child.killed) {
        if (process.platform === 'win32') {
          spawn('taskkill', ['/pid', child.pid.toString(), '/f', '/t']);
        } else {
          child.kill('SIGTERM');
        }
      }
    } catch {}
  };

  process.on('exit', cleanup);
  process.on('SIGINT', () => { cleanup(); process.exit(1); });
  process.on('SIGTERM', () => { cleanup(); process.exit(1); });

  try {
    const healthUrl = `http://127.0.0.1:${port}/health`;
    const t0 = Date.now();
    await waitForHealth(healthUrl);
    console.log(`[quick-check] server ready in ${((Date.now() - t0) / 1000).toFixed(2)}s`);

    const apiUrl = `http://127.0.0.1:${port}/v1/systemone`;
    let passed = 0;
    const latencies = [];

    console.log(`\nRunning 10 test triage questions:\n`);

    let responsesOk = 0;
    for (let i = 0; i < CASES.length; i++) {
      const [prompt, expected] = CASES[i];
      const reqStart = Date.now();

      const res = await fetch(apiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ state: prompt, questions: QUESTIONS })
      });

      const reqDur = Date.now() - reqStart;
      latencies.push(reqDur);

      if (!res.ok) {
        const errText = await res.text();
        console.log(`[${i + 1}/10] ❌ HTTP ERROR ${res.status}: ${errText}`);
        continue;
      }

      const data = await res.json();
      const choice = data.answers?.department?.choice;
      const prob = data.answers?.department?.probabilities?.[choice] || 0;
      responsesOk++;
      const isMatch = choice === expected;

      console.log(
        `[${i + 1}/10] ${choice ? '✔ OK' : '❌ NO CHOICE'} | got: ${String(choice).padEnd(8)} | exp: ${expected.padEnd(8)} ${isMatch ? '(match)' : '       '} | p=${(prob * 100).toFixed(1)}% | ${reqDur}ms | "${prompt.slice(0, 38)}..."`
      );
    }

    const totalDur = latencies.reduce((a, b) => a + b, 0);
    const avgLatency = (totalDur / latencies.length).toFixed(1);
    const rps = (1000 / (totalDur / latencies.length)).toFixed(2);

    // Multi-question request: exercises the K>1 path (several questions in a
    // single request). Every question must come back with a choice and a
    // normalized probability distribution.
    console.log(`\nRunning multi-question request (3 questions, 1 request):\n`);
    const multiQs = {
      department: QUESTIONS.department,
      priority: {
        type: 'choice',
        instructions: 'How urgent is this request?',
        criteria: {
          low: 'can wait a few business days with no impact',
          normal: 'should be handled this week',
          high: 'blocks the customer right now and needs attention today',
        },
      },
      sentiment: {
        type: 'choice',
        instructions: 'What is the customer sentiment?',
        criteria: { calm: 'neutral or friendly tone', upset: 'angry, frustrated or disappointed' },
      },
    };
    const multiStart = Date.now();
    let multiOk = false;
    try {
      const res = await fetch(apiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ state: CASES[0][0], questions: multiQs })
      });
      const multiDur = Date.now() - multiStart;
      if (!res.ok) {
        console.log(`[multi]  ❌ HTTP ERROR ${res.status}: ${await res.text()}`);
      } else {
        const data = await res.json();
        const keys = Object.keys(multiQs);
        const bad = keys.filter((k) => {
          const a = data.answers?.[k];
          if (!a?.choice) return true;
          const sum = Object.values(a.probabilities || {}).reduce((x, y) => x + y, 0);
          return Math.abs(sum - 1) > 0.01;
        });
        multiOk = bad.length === 0;
        const summary = keys.map((k) => `${k}=${data.answers?.[k]?.choice}`).join(' ');
        console.log(
          `[multi]  ${multiOk ? '✔ OK' : '❌ INVALID'} | ${summary} | ${multiDur}ms`
        );
      }
    } catch (err) {
      console.log(`[multi]  ❌ REQUEST FAILED: ${err}`);
    }

    console.log(`\n==================================================`);
    console.log(`Performance & Health Check Summary:`);
    console.log(`Successful Inferences: ${responsesOk}/${CASES.length}`);
    console.log(`Average Latency     : ${avgLatency} ms`);
    console.log(`Throughput          : ${rps} req/s`);
    console.log(`==================================================\n`);

    cleanup();

    if (responsesOk === 0) {
      console.error(`[quick-check] FAILED: server did not produce valid inferences.`);
      process.exit(1);
    }
    if (!multiOk) {
      console.error(`[quick-check] FAILED: multi-question request did not return valid answers for all questions.`);
      process.exit(1);
    }

    console.log(`[quick-check] Platform smoke check passed successfully! ✔`);
    process.exit(0);
  } catch (err) {
    cleanup();
    console.error(`[quick-check] ERROR:`, err);
    process.exit(1);
  }
}

main();
