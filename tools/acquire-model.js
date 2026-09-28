#!/usr/bin/env node
/**
 * acquire-model.js — downloads and extracts `models/qwen-0.8b-q4.model`
 * from GitHub releases if the ONNX files are not already present.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { execSync } from 'node:child_process';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');
const MODELS_DIR = path.join(ROOT, 'models');
const MANIFEST_PATH = path.join(MODELS_DIR, 'model.manifest.json');

const REQUIRED_FILES = [
  'embed_tokens_q4.onnx',
  'embed_tokens_q4.onnx_data',
  'decoder_model_merged_q4.onnx',
  'decoder_model_merged_q4.onnx_data',
  'tokenizer.json'
];

async function sha256(filePath) {
  const hash = createHash('sha256');
  const stream = fs.createReadStream(filePath);
  for await (const chunk of stream) hash.update(chunk);
  return hash.digest('hex');
}

async function main() {
  const allPresent = REQUIRED_FILES.every((f) => fs.existsSync(path.join(MODELS_DIR, f)));
  if (allPresent) {
    console.log('[acquire] all ONNX model files are already present.');
    return;
  }

  const manifest = JSON.parse(fs.readFileSync(MANIFEST_PATH, 'utf8'));
  const modelArchive = path.join(MODELS_DIR, manifest.model);

  if (!fs.existsSync(modelArchive)) {
    console.log(`[acquire] downloading ${manifest.model} from ${manifest.source}...`);
    let downloaded = false;
    try {
      execSync(`curl -sL -f --retry 3 --retry-delay 2 -o "${modelArchive}.tmp" "${manifest.source}"`, { stdio: 'inherit' });
      fs.renameSync(`${modelArchive}.tmp`, modelArchive);
      downloaded = true;
    } catch {}

    if (!downloaded) {
      const { pipeline } = await import('node:stream/promises');
      const { Readable } = await import('node:stream');
      const resp = await fetch(manifest.source, {
        headers: { 'user-agent': 'qwen-system-one-acquire' },
        redirect: 'follow'
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching model archive: ${resp.statusText}`);
      const fileStream = fs.createWriteStream(`${modelArchive}.tmp`);
      await pipeline(Readable.fromWeb(resp.body), fileStream);
      fs.renameSync(`${modelArchive}.tmp`, modelArchive);
    }
  }

  console.log('[acquire] verifying model archive sha256...');
  const actualHash = await sha256(modelArchive);
  if (actualHash !== manifest.sha256) {
    throw new Error(`SHA256 mismatch! Expected ${manifest.sha256}, got ${actualHash}`);
  }
  console.log('[acquire] sha256 verified ✔');

  console.log('[acquire] extracting model archive...');
  execSync(`tar -xzf "${modelArchive}" -C "${MODELS_DIR}"`, { stdio: 'inherit' });
  console.log('[acquire] extraction complete ✔');
}

main().catch((err) => {
  console.error('[acquire] FAILED:', err.message);
  process.exit(1);
});
