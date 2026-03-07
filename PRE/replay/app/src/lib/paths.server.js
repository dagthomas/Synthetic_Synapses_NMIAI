import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

// Project root: PRE/
export const PROJECT_ROOT = resolve(__dirname, '..', '..', '..', '..');

// Sibling project directories
export const ZIG_BOT_DIR = resolve(PROJECT_ROOT, 'grocery-bot-zig');
export const GPU_DIR = resolve(PROJECT_ROOT, 'grocery-bot-gpu');
export const B200_DIR = resolve(PROJECT_ROOT, 'grocery-bot-b200');
export const REPLAY_DIR = resolve(PROJECT_ROOT, 'replay');
