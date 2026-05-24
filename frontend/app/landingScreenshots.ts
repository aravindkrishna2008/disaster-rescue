/** Committed assets under frontend/public/images — verified at build time. */
export const LANDING_SCREENSHOTS = {
  gym: {
    src: '/images/training-gym.png',
    alt: 'Training Gym dashboard showing eight disaster environment cards with 3D rollouts',
    caption: 'Mission Control — train PPO, run all eight gym environments, inspect reach metrics',
  },
  generate: {
    src: '/images/scene-generator.png',
    alt: 'Scene Generator output with 3D preview, EvalAgent score, and generated scene details',
    caption:
      'Describe a disaster in plain language — Gemini builds the layout, EvalAgent validates, 3D preview loads',
  },
  console: {
    src: '/images/interactive-console.png',
    alt: 'Interactive Console showing generated scene response with tactical view, rollout GIF, and episode telemetry',
    caption:
      'Generated scene response — tactical route, rendered rollout, episode telemetry, and EvalAgent validation',
  },
} as const;

export const LANDING_SCREENSHOT_FILES = [
  'public/images/training-gym.png',
  'public/images/scene-generator.png',
  'public/images/interactive-console.png',
] as const;
