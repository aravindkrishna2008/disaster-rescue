export type BenchmarkScene = {
  id: string;
  name: string;
  difficulty: 'medium' | 'hard';
  hazards: number;
  buried: boolean;
  signal?: string;
  objective: string;
};

export const BENCHMARK_SCENES: BenchmarkScene[] = [
  {
    id: '01',
    name: 'earthquake_corridor',
    difficulty: 'medium',
    hazards: 2,
    buried: false,
    objective: 'Navigate collapsed corridor debris to reach visible survivor.',
  },
  {
    id: '02',
    name: 'flooded_hospital',
    difficulty: 'hard',
    hazards: 2,
    buried: false,
    objective: 'Cross flooded ward layout with tight obstacle spacing.',
  },
  {
    id: '03',
    name: 'wildfire_shelter',
    difficulty: 'medium',
    hazards: 2,
    buried: false,
    objective: 'Route around fire hazard zones to sheltered survivor.',
  },
  {
    id: '04',
    name: 'collapsed_bridge',
    difficulty: 'hard',
    hazards: 2,
    buried: false,
    objective: 'Traverse bridge collapse geometry with diagonal clearance.',
  },
  {
    id: '05',
    name: 'chemical_plant',
    difficulty: 'medium',
    hazards: 2,
    buried: false,
    objective: 'Minimize hazard exposure while closing distance in plant sector.',
  },
  {
    id: '06',
    name: 'downtown_rubble',
    difficulty: 'medium',
    hazards: 2,
    buried: true,
    signal: 'thermal_audio',
    objective: 'Detect buried survivor under rubble, then reach within 0.75 m.',
  },
  {
    id: '07',
    name: 'parking_garage_pancake',
    difficulty: 'hard',
    hazards: 3,
    buried: true,
    signal: 'acoustic_vibration',
    objective: 'Pancake-collapse slab cover; acoustic detection before reach.',
  },
  {
    id: '08',
    name: 'aftershock_triage_maze',
    difficulty: 'hard',
    hazards: 3,
    buried: true,
    signal: 'thermal_audio',
    objective: 'Multi-hazard maze with buried triage target.',
  },
];

export type ObsComponent = {
  index: string;
  dims: number;
  symbol: string;
  description: string;
};

export const OBSERVATION_SPEC: ObsComponent[] = [
  { index: '0–1', dims: 2, symbol: 'p_r', description: 'Robot position (x, y) in world frame' },
  { index: '2–3', dims: 2, symbol: 'p_s', description: 'Survivor absolute position (x, y)' },
  { index: '4–5', dims: 2, symbol: 'Δp', description: 'Relative vector from robot to survivor' },
  { index: '6', dims: 1, symbol: 'd', description: 'Euclidean distance to survivor' },
  { index: '7–8', dims: 2, symbol: 'v', description: 'Per-step displacement (proxy velocity)' },
  { index: '9–17', dims: 9, symbol: 'O', description: 'Nearest 3 obstacles: (Δx, Δy, clearance) each' },
  { index: '18–20', dims: 3, symbol: 'D', description: 'Buried flag, detection flag, signal strength' },
];

export type RewardTerm = {
  term: string;
  formula: string;
  notes: string;
};

export const REWARD_TERMS: RewardTerm[] = [
  { term: 'Progress shaping', formula: '+20 × (d_{t-1} − d_t)', notes: 'Dense incentive toward survivor' },
  { term: 'Distance penalty', formula: '−0.01 × d_t', notes: 'Penalizes lingering far from goal' },
  { term: 'Step cost', formula: '−0.02', notes: 'Fixed per-step budget pressure' },
  { term: 'Collision', formula: '−8.0', notes: 'Obstacle or terrain block contact' },
  { term: 'Hazard zone', formula: '−1.0', notes: 'Inside circular chemical/fire hazard' },
  { term: 'Terrain roughness', formula: '−0.45 × roughness', notes: 'Procedural heightfield friction' },
  { term: 'Terrain danger', formula: '−1.75 × danger', notes: 'High-risk terrain cells' },
  { term: 'First detection', formula: '+5.0', notes: 'One-time bonus when buried survivor detected' },
  { term: 'Reach success', formula: '+200.0', notes: 'Terminal bonus when d < 0.75 m' },
];

export type ResearchTask = {
  id: string;
  area: string;
  status: 'complete' | 'in_progress' | 'blocked' | 'planned';
  description: string;
};

export const RESEARCH_TASKS: ResearchTask[] = [
  {
    id: 'T-01',
    area: 'Data pipeline',
    status: 'complete',
    description: 'Export runs/ manifests with TensorBoard curves, eval JSON, and episode rollouts.',
  },
  {
    id: 'T-02',
    area: 'Buried detection',
    status: 'complete',
    description: '21D observation with detection features; 3 buried benchmark scenes.',
  },
  {
    id: 'T-03',
    area: 'Terrain generalization',
    status: 'blocked',
    description: 'Retrain PPO after procedural terrain merge — current policy 0/8 on held-out eval.',
  },
  {
    id: 'T-04',
    area: 'Replay API',
    status: 'complete',
    description: 'Episode loader + deterministic replay endpoints for BC warm-start.',
  },
  {
    id: 'T-05',
    area: 'Mission UI',
    status: 'in_progress',
    description: 'Replay dropdown and live eval persistence in mission-control dashboard.',
  },
  {
    id: 'T-06',
    area: 'Gemini grounding',
    status: 'complete',
    description: 'Gemini 3.5 Flash goal selection and NL scene generation via ScenarioAgent.',
  },
];

export const MDP_CONSTANTS = {
  actionSpace: 'ℝ² — clipped unit disk, scaled by 0.18 m/step',
  obsDim: 21,
  maxStepsTrain: 600,
  maxStepsEval: 300,
  reachTolerance: '0.75 m',
  worldBounds: '±8.0 m (16 m × 16 m)',
  parallelEnvs: 8,
  algorithm: 'PPO (Stable-Baselines3)',
  policy: 'MLP 256 × 256',
  trainingSteps: '500,000',
};
