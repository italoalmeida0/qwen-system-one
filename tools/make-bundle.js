#!/usr/bin/env node
// tools/make-bundle.js — packs bin/<plat>-<arch>-musl/{qwen-serve,lib/} into ONE
// self-extracting file: bin/<plat>-<arch>-musl/qwen-serve.bundle
//
// Layout of the bundle:
//   [loader shell script][\n__PAYLOAD__\n][tar.gz of {qwen-serve,lib/}]
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(__dirname, '..');

const LOADER = `#!/bin/sh
# qwen-serve self-extracting bundle (musl). No system deps beyond musl.
set -e
BUNDLE="$0"
MARK=$(grep -a -n '^__PAYLOAD__$' "$BUNDLE" | tail -n 1 | cut -d: -f1)
H=$(sha256sum "$BUNDLE" 2>/dev/null | cut -d' ' -f1 || shasum -a 256 "$BUNDLE" | cut -d' ' -f1)
try_extract() {
  rm -rf "$1"; mkdir -p "$1" 2>/dev/null || return 1
  tail -n +$((MARK + 1)) "$BUNDLE" | tar -xz -C "$1" 2>/dev/null || return 1
  chmod +x "$1/qwen-serve" 2>/dev/null
  LDD_PATH="$1/lib:$LD_LIBRARY_PATH"
  LD_LIBRARY_PATH="$LDD_PATH" "$1/qwen-serve" --help >/dev/null 2>&1 || return 1
  touch "$1/.ok" 2>/dev/null || return 1
  return 0
}
DEST=""
for BASE in "\${TMPDIR:-/dev/shm}/qwen" "\${XDG_CACHE_HOME:-$HOME/.cache}/qwen"; do
  CAND="$BASE/$H"
  if [ -f "$CAND/.ok" ]; then DEST="$CAND"; break; fi
  if try_extract "$CAND"; then DEST="$CAND"; break; fi
done
if [ -z "$DEST" ]; then echo "qwen-serve: cannot extract bundle" >&2; exit 1; fi
export LD_LIBRARY_PATH="$DEST/lib:\${LD_LIBRARY_PATH:-}"
exec "$DEST/qwen-serve" "$@"
__PAYLOAD__
`;

function bundleOne(dir) {
  const srcDir = path.join(root, 'dist', 'bin', dir);
  const bin = path.join(srcDir, 'qwen-serve');
  const lib = path.join(srcDir, 'lib');
  if (!fs.existsSync(bin) || !fs.existsSync(lib)) {
    console.log(`skip ${dir}: no musl layout`);
    return;
  }
  const out = path.join(srcDir, 'qwen-serve.bundle');
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'qwenbundle-'));
  fs.copyFileSync(bin, path.join(tmp, 'qwen-serve'));
  execFileSync('tar', ['-czf', path.join(tmp, 'payload.tgz'), '-C', srcDir, 'qwen-serve', 'lib']);
  const payload = fs.readFileSync(path.join(tmp, 'payload.tgz'));
  const loaderBuf = Buffer.from(LOADER.replace(/\r\n/g, '\n'), 'utf8');
  fs.writeFileSync(out, Buffer.concat([loaderBuf, payload]));
  fs.chmodSync(out, 0o755);
  const hash = createHash('sha256').update(fs.readFileSync(out)).digest('hex').slice(0, 12);
  console.log(`${dir}: bundle ${(fs.statSync(out).size / 1048576).toFixed(1)}MB sha=${hash}`);
  fs.rmSync(tmp, { recursive: true, force: true });
}

const dirs = process.argv.slice(2);
(dirs.length ? dirs : ['linux-x64-musl', 'linux-arm64-musl']).forEach(bundleOne);
