export type CurvePoint = { step: number; reward: number };

export type AdvancedStat = {
  label: string;
  value: string;
  group?: string;
};

export type TrainingRun = {
  id: string;
  name: string;
  subtitle: string;
  totalSteps: number;
  solvedThreshold: number;
  yMin: number;
  yMax: number;
  caption: string;
  curve: CurvePoint[];
  checkpoints: string[];
  fileTree: string[];
  advancedStats: AdvancedStat[];
};

export const TRAINING_RUNS: TrainingRun[] = [
  {
    id: 'ppo_fixed_six',
    name: 'ppo_fixed_six',
    subtitle: '6-scene vectorized PPO · production policy',
    totalSteps: 500_000,
    solvedThreshold: 0,
    yMin: -250,
    yMax: 50,
    caption: 'Mean episode reward trends positive and converges near step 360k.',
    curve: [
      { step: 0, reward: -250 },
      { step: 40_000, reward: -215 },
      { step: 80_000, reward: -175 },
      { step: 120_000, reward: -145 },
      { step: 180_000, reward: -110 },
      { step: 240_000, reward: -78 },
      { step: 300_000, reward: -42 },
      { step: 360_000, reward: -8 },
      { step: 420_000, reward: 28 },
      { step: 480_000, reward: 38 },
      { step: 500_000, reward: 32 },
    ],
    checkpoints: [
      'ppo_fixed_six_120000_steps.zip',
      'ppo_fixed_six_240000_steps.zip',
      'ppo_fixed_six_360000_steps.zip',
      'ppo_fixed_six_480000_steps.zip',
      'ppo_fixed_six_final.zip',
    ],
    fileTree: [
      'runs/',
      '  └── ppo_fixed_six_final/',
      '      ├── summary.json          # Run parameters & meta',
      '      ├── eval.json             # 6-scene test results',
      '      ├── checkpoints/          # Models at step checkpoints',
      '      │   ├── ppo_fixed_six_120000_steps.zip',
      '      │   ├── ppo_fixed_six_360000_steps.zip',
      '      │   └── ppo_fixed_six_480000_steps.zip',
      '      └── tb_logs/              # TensorBoard diagnostics',
    ],
    advancedStats: [
      { group: 'Run', label: 'Algorithm', value: 'PPO (Proximal Policy Optimization)' },
      { group: 'Run', label: 'Policy network', value: 'MLP 256 × 256' },
      { group: 'Run', label: 'Total timesteps', value: '500,000' },
      { group: 'Run', label: 'Parallel envs', value: '6 (one per disaster scene)' },
      { group: 'Run', label: 'Checkpoint interval', value: '20,000 steps' },
      { group: 'Hyperparameters', label: 'Batch size', value: '256' },
      { group: 'Hyperparameters', label: 'n_steps', value: '2,048' },
      { group: 'Hyperparameters', label: 'Gamma (γ)', value: '0.99' },
      { group: 'Hyperparameters', label: 'GAE λ', value: '0.95' },
      { group: 'Hyperparameters', label: 'Clip range', value: '0.2' },
      { group: 'Hyperparameters', label: 'Entropy coef', value: '0.0' },
      { group: 'Hyperparameters', label: 'Learning rate', value: '3e-4 (SB3 default)' },
      { group: 'Evaluation', label: 'Scenes evaluated', value: '6 fixed disaster layouts' },
      { group: 'Evaluation', label: 'Reach tolerance', value: '0.75 m' },
      { group: 'Evaluation', label: 'Step budget (eval)', value: '150' },
      { group: 'Evaluation', label: 'Success @ final checkpoint', value: '5 / 6 scenes' },
      { group: 'Evaluation', label: 'Mean final distance', value: '0.41 m' },
      { group: 'Evaluation', label: 'Est. convergence step', value: '~360k' },
      { group: 'Evaluation', label: 'Best checkpoint', value: '480k steps' },
    ],
  },
  {
    id: 'ppo_disaster',
    name: 'ppo_disaster',
    subtitle: 'Early baseline · single-env exploration',
    totalSteps: 120_000,
    solvedThreshold: 0,
    yMin: -250,
    yMax: 50,
    caption: 'Baseline run — reward improves steadily but does not reach solved threshold within 120k steps.',
    curve: [
      { step: 0, reward: -250 },
      { step: 20_000, reward: -220 },
      { step: 40_000, reward: -185 },
      { step: 60_000, reward: -148 },
      { step: 80_000, reward: -115 },
      { step: 100_000, reward: -82 },
      { step: 120_000, reward: -58 },
    ],
    checkpoints: ['ppo_disaster_60000_steps.zip', 'ppo_disaster_120000_steps.zip'],
    fileTree: [
      'runs/',
      '  └── ppo_disaster/',
      '      ├── summary.json',
      '      ├── eval.json',
      '      ├── checkpoints/',
      '      │   ├── ppo_disaster_60000_steps.zip',
      '      │   └── ppo_disaster_120000_steps.zip',
      '      └── tb_logs/',
    ],
    advancedStats: [
      { group: 'Run', label: 'Algorithm', value: 'PPO' },
      { group: 'Run', label: 'Policy network', value: 'MLP 256 × 256' },
      { group: 'Run', label: 'Total timesteps', value: '120,000' },
      { group: 'Run', label: 'Parallel envs', value: '1' },
      { group: 'Run', label: 'Checkpoint interval', value: '20,000 steps' },
      { group: 'Hyperparameters', label: 'Batch size', value: '256' },
      { group: 'Hyperparameters', label: 'Gamma (γ)', value: '0.99' },
      { group: 'Hyperparameters', label: 'GAE λ', value: '0.95' },
      { group: 'Evaluation', label: 'Scenes evaluated', value: '1 (warehouse prototype)' },
      { group: 'Evaluation', label: 'Success @ 120k', value: '0 / 1 (in progress)' },
      { group: 'Evaluation', label: 'Mean final distance', value: '1.82 m' },
      { group: 'Evaluation', label: 'Notes', value: 'Superseded by ppo_fixed_six multi-scene run' },
    ],
  },
];

export function formatSteps(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`.replace('.0M', 'M');
  if (n >= 1_000) return `${Math.round(n / 1_000)}k`;
  return String(n);
}

export function stepToX(step: number, maxStep: number, xMin = 30, xMax = 380): number {
  return xMin + (step / maxStep) * (xMax - xMin);
}

export function rewardToY(reward: number, yMin: number, yMax: number, yBottom = 170, yTop = 15): number {
  const pct = (reward - yMin) / (yMax - yMin);
  return yBottom - pct * (yBottom - yTop);
}

export function curveToPath(
  curve: CurvePoint[],
  maxStep: number,
  yMin: number,
  yMax: number,
): string {
  if (curve.length === 0) return '';
  return curve
    .map((p, i) => {
      const x = stepToX(p.step, maxStep).toFixed(1);
      const y = rewardToY(p.reward, yMin, yMax).toFixed(1);
      return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
    })
    .join(' ');
}

export function xAxisTicks(maxStep: number, count = 5): number[] {
  const raw = Array.from({ length: count }, (_, i) => Math.round((maxStep / (count - 1)) * i));
  return [...new Set(raw)];
}

export function yAxisTicks(yMin: number, yMax: number, count = 4): number[] {
  return Array.from({ length: count }, (_, i) => Math.round(yMin + ((yMax - yMin) / (count - 1)) * i));
}
