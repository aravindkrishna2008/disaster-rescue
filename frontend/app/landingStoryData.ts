import { LANDING_SCREENSHOTS } from './landingScreenshots';

export type BuildPhase = {
  id: string;
  index: string;
  title: string;
  headline: string;
  story: string[];
  deliverables: string[];
  /** Exported run manifest for live stats (Training Gym policy) */
  runId?: string;
  screenshot?: { src: string; alt: string; caption: string };
  href: string;
  linkLabel: string;
  showCta?: boolean;
};

export const BUILD_PHASES: BuildPhase[] = [
  {
    id: 'phase-gym',
    index: '01',
    title: 'Training Gym',
    headline: 'Stress-test policies before they touch a new disaster layout.',
    story: [
      'Rescue robots fail when training data does not match real rubble. The Training Gym runs PPO across eight fixed MuJoCo disaster scenes — corridors, chemical plants, buried-rubble layouts — so you can train, evaluate, and compare checkpoints on the same benchmark every time.',
      'Run single episodes or queue all eight in parallel, watch reach rate and reward live, and export manifests with learning curves and rollout GIFs. That closed loop is what lets you iterate on locomotion and navigation without re-writing the sim by hand.',
    ],
    deliverables: [
      'Vectorized PPO training on 8 shared-weight environments',
      'Per-scene eval rollouts with reach / detection metrics',
      'Run All queue with concurrency limits for stable parallel eval',
      'Exported runs/ artifacts — curves, episode JSON, GIFs',
    ],
    runId: 'g1_locomotion_walk_final',
    screenshot: LANDING_SCREENSHOTS.gym,
    href: '/mission-control',
    linkLabel: 'Enter Training Gym',
  },
  {
    id: 'phase-generate',
    index: '02',
    title: 'Scene Generator',
    headline: 'New disaster layouts without hand-authoring every obstacle.',
    story: [
      'Fixed benchmarks catch regressions, but robotics needs generalization. Describe a disaster in plain language — “collapsed hospital wing with gas leak near triage” — and Gemini 3.5 Flash returns a validated scene: survivor positions, hazard zones, rubble assets, and a MuJoCo-ready layout through ScenarioAgent.',
      'You preview the environment in 3D, then run the current policy against it immediately. That shortens the loop from “what if this building looked different?” to a measurable rollout, which is how you test whether a rescue stack transfers beyond the eight training scenes.',
    ],
    deliverables: [
      'NL → structured scene JSON via Gemini 3.5 Flash',
      'Asset catalog + scene_adapter validation before sim load',
      'Live 3D preview and optional instant episode rollout',
      'Candidate scenes for curriculum and domain randomization',
    ],
    screenshot: LANDING_SCREENSHOTS.generate,
    href: '/generate',
    linkLabel: 'Generate a Scene',
    showCta: false,
  },
  {
    id: 'phase-console',
    index: '03',
    title: 'Interactive Console',
    headline: 'Operators speak; the robot picks who to save first.',
    story: [
      'Field rescue is not fully autonomous — a human still sets priorities. The Interactive Console is the human-in-the-loop layer: type “save the child first” or “ignore the adult, triage the critical case,” and Gemini 3.5 Flash maps that order to a survivor target while a trained policy handles navigation in MuJoCo.',
      'The console surfaces Gemini’s reasoning, live trajectory, and reach/timeout outcomes so operators and researchers can see language grounding drive physical motion. That is the demo loop for language-conditioned rescue robotics: NL intent → structured goal → policy execution → auditable result.',
    ],
    deliverables: [
      'POST /command — NL order → Gemini target_id + reason',
      'PPO rollout to selected survivor with step budget',
      'Live telemetry log — heuristic parse, Gemini correction, result',
      'Dual-survivor prioritization for triage scenarios',
    ],
    screenshot: LANDING_SCREENSHOTS.console,
    href: '/console',
    linkLabel: 'Open Interactive Console',
    showCta: false,
  },
];
