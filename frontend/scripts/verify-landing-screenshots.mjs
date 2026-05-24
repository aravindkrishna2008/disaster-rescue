import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const frontendRoot = join(dirname(fileURLToPath(import.meta.url)), '..');

const required = [
  'public/images/training-gym.png',
  'public/images/scene-generator.png',
  'public/images/interactive-console.png',
];

const missing = required.filter((rel) => !existsSync(join(frontendRoot, rel)));

if (missing.length > 0) {
  console.error('Missing landing page screenshots (commit these under frontend/public/images/):');
  for (const rel of missing) console.error(`  - ${rel}`);
  process.exit(1);
}

console.log(`Landing screenshots OK (${required.length} files)`);
