import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { execSync } from 'child_process';

const __dirname = dirname(fileURLToPath(import.meta.url));

// Project root: PRE/
export const PROJECT_ROOT = resolve(__dirname, '..', '..', '..', '..');

// Sibling project directories
export const ZIG_BOT_DIR = resolve(PROJECT_ROOT, 'grocery-bot-zig');
export const GPU_DIR = resolve(PROJECT_ROOT, 'grocery-bot-gpu');
export const B200_DIR = resolve(PROJECT_ROOT, 'grocery-bot-b200');
export const REPLAY_DIR = resolve(PROJECT_ROOT, 'replay');

// Resolve full python path (spawn fails with bare 'python' on Windows Vite dev server)
let _python;
try {
	const raw = execSync('where python', { encoding: 'utf8' }).trim();
	_python = raw.split(/\r?\n/)[0].replace(/\\/g, '/');
	console.log('[paths] PYTHON resolved to:', _python);
} catch (e) {
	console.log('[paths] where python failed:', e.message);
	_python = 'python';
}
export const PYTHON = _python;
