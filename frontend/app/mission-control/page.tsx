'use client';

import Link from 'next/link';
import { useCallback, useEffect, useRef, useState } from 'react';
import WorkflowNav from '../WorkflowNav';

type Scene = {
  index: number;
  name: string;
  difficulty: string;
  robot_start: number[];
  survivor_pos: number[];
  survivor?: SurvivorMeta;
  obstacle_count: number;
  hazard_count: number;
  obstacles?: Obstacle[];
  hazards?: Hazard[];
  terrain?: Terrain;
};

type Obstacle = { pos: number[]; size: number[]; color?: string };
type Hazard = { center: number[]; radius: number; type?: string; color?: string };
type Terrain = {
  grid_size: number;
  heights: number[][];
  roughness?: number[][];
  danger?: number[][];
  rigid?: number[][];
};

type SurvivorMeta = {
  buried?: boolean;
  cover?: string;
  detection_radius?: number;
  signal?: string;
};

type DetectionEvent = {
  step: number;
  robot_pos: number[];
  signal?: string;
  radius?: number;
};

type RunResult = {
  scene_index: number;
  scene_name: string;
  difficulty: string;
  reached: boolean;
  steps: number;
  total_reward: number;
  gif_url: string;
  max_steps: number;
  survivor?: SurvivorMeta;
  detection_event?: DetectionEvent | null;
  trajectory?: number[][];
};

type RunState =
  | { status: 'idle' }
  | { status: 'queued' }
  | { status: 'running'; startedAt: number; controller: AbortController }
  | { status: 'done'; result: RunResult; finishedAt: number; cacheBust: number }
  | { status: 'error'; message: string }
  | { status: 'cancelled' };

type StepPreset = { label: string; value: number };

const RUN_ALL_CONCURRENCY = 2;

const FALLBACK_PRESETS: StepPreset[] = [
  { label: '50k', value: 50_000 },
  { label: '100k', value: 100_000 },
  { label: '200k', value: 200_000 },
  { label: '500k', value: 500_000 },
  { label: '1M', value: 1_000_000 },
];

const FIXED_NUM_ENVS = 8;

type TrainStatus = {
  status: 'idle' | 'running' | 'done' | 'error' | 'cancelled';
  total_steps: number;
  started_at: number | null;
  finished_at: number | null;
  pid: number | null;
  exit_code: number | null;
  error: string | null;
};

const prettyName = (n: string) =>
  n.split('_').map((w) => w[0].toUpperCase() + w.slice(1)).join(' ');

const tsNow = () => {
  const d = new Date();
  return [d.getUTCHours(), d.getUTCMinutes(), d.getUTCSeconds()]
    .map((v) => String(v).padStart(2, '0'))
    .join(':');
};

const ARENA_HALF = 8;

function planarDistance(a: number[], b: number[]) {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function getRouteStats(scene: Scene, result: RunResult) {
  const points = [scene.robot_start, ...(result.trajectory ?? [])];
  const travelled = points.slice(1).reduce(
    (sum, point, index) => sum + planarDistance(points[index], point),
    0,
  );
  const direct = planarDistance(scene.robot_start, scene.survivor_pos);
  const finalPosition = points[points.length - 1] ?? scene.robot_start;
  const finalDistance = planarDistance(finalPosition, scene.survivor_pos);
  const efficiency = travelled > 0 ? Math.min(100, (direct / travelled) * 100) : 0;
  const detectionLead = result.detection_event
    ? Math.max(0, result.steps - result.detection_event.step)
    : null;
  return { points, travelled, direct, finalDistance, efficiency, detectionLead };
}

function RouteGraph({ scene, result }: { scene: Scene; result: RunResult }) {
  const width = 252;
  const height = 116;
  const pad = 10;
  const scaleX = (x: number) => pad + ((x + ARENA_HALF) / (ARENA_HALF * 2)) * (width - pad * 2);
  const scaleY = (y: number) => height - pad - ((y + ARENA_HALF) / (ARENA_HALF * 2)) * (height - pad * 2);
  const { points } = getRouteStats(scene, result);
  const trace = points.map((p) => `${scaleX(p[0]).toFixed(1)},${scaleY(p[1]).toFixed(1)}`).join(' ');
  const detection = result.detection_event?.robot_pos;

  return (
    <div className="mc-route-graph">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`Route trace for ${scene.name}`}>
        <path d={`M ${width / 2} ${pad} V ${height - pad} M ${pad} ${height / 2} H ${width - pad}`} className="mc-route-grid" />
        {scene.terrain?.heights?.map((row, rowIdx) =>
          row.map((heightValue, colIdx) => {
            const gridSize = scene.terrain?.grid_size || row.length;
            const danger = scene.terrain?.danger?.[rowIdx]?.[colIdx] ?? 0;
            const roughness = scene.terrain?.roughness?.[rowIdx]?.[colIdx] ?? 0;
            if (heightValue <= 0.01 && roughness <= 0.05 && danger <= 0) return null;
            const cellW = (width - pad * 2) / gridSize;
            const cellH = (height - pad * 2) / gridSize;
            const fill = danger > 0.7 ? '#c93d2d' : roughness > 0.85 ? '#8a7a52' : '#6f8a67';
            return (
              <rect
                key={`terrain-${rowIdx}-${colIdx}`}
                x={pad + colIdx * cellW}
                y={pad + rowIdx * cellH}
                width={cellW}
                height={cellH}
                fill={fill}
                opacity={danger > 0 ? 0.22 : 0.13}
              />
            );
          })
        )}
        {(scene.hazards ?? []).map((hazard, idx) => (
          <circle
            key={`haz-${idx}`}
            cx={scaleX(hazard.center[0])}
            cy={scaleY(hazard.center[1])}
            r={(hazard.radius / (ARENA_HALF * 2)) * (width - pad * 2)}
            className="mc-route-hazard"
            style={{ stroke: hazard.color, fill: hazard.color }}
          />
        ))}
        {(scene.obstacles ?? []).map((obstacle, idx) => (
          <rect
            key={`obs-${idx}`}
            x={scaleX(obstacle.pos[0] - obstacle.size[0])}
            y={scaleY(obstacle.pos[1] + obstacle.size[1])}
            width={(obstacle.size[0] / ARENA_HALF) * (width - pad * 2)}
            height={(obstacle.size[1] / ARENA_HALF) * (height - pad * 2)}
            className="mc-route-obstacle"
            style={{ stroke: obstacle.color, fill: obstacle.color }}
          />
        ))}
        <polyline points={trace} className="mc-route-trace" />
        <circle cx={scaleX(scene.robot_start[0])} cy={scaleY(scene.robot_start[1])} r="3.5" className="mc-route-start" />
        <circle cx={scaleX(scene.survivor_pos[0])} cy={scaleY(scene.survivor_pos[1])} r="4" className="mc-route-target" />
        {detection && (
          <circle cx={scaleX(detection[0])} cy={scaleY(detection[1])} r="4.5" className="mc-route-detect" />
        )}
      </svg>
      <div className="mc-route-key">
        <span className="trace">route</span>
        <span className="target">survivor</span>
        {detection && <span className="detect">sensor lock</span>}
      </div>
    </div>
  );
}

export default function MissionControlPage() {
  const [clockUtc, setClockUtc] = useState('--:--:--');

  useEffect(() => {
    setClockUtc(tsNow());
    const id = setInterval(() => setClockUtc(tsNow()), 1000);
    return () => clearInterval(id);
  }, []);

  const [scenes, setScenes] = useState<Scene[]>([]);
  const [runs, setRuns] = useState<Record<number, RunState>>({});
  const [trainSteps, setTrainSteps] = useState(200_000);
  const [presets, setPresets] = useState<StepPreset[]>(FALLBACK_PRESETS);
  const [defaultMaxSteps, setDefaultMaxSteps] = useState(300);
  const [bootError, setBootError] = useState<string | null>(null);
  const [killing, setKilling] = useState(false);
  const [train, setTrain] = useState<TrainStatus>({
    status: 'idle',
    total_steps: 0,
    started_at: null,
    finished_at: null,
    pid: null,
    exit_code: null,
    error: null,
  });
  const killedRef = useRef(false);
  const runsRef = useRef<Record<number, RunState>>({});
  const runAllQueueRef = useRef<number[]>([]);
  const runAllActiveRef = useRef(0);
  runsRef.current = runs;

  const [trainingElapsed, setTrainingElapsed] = useState(0);

  useEffect(() => {
    if (train.status !== 'running' || !train.started_at) {
      setTrainingElapsed(0);
      return;
    }
    const getElapsed = () => Math.max(0, Math.floor(Date.now() / 1000 - (train.started_at ?? (Date.now() / 1000))));
    setTrainingElapsed(getElapsed());
    const id = setInterval(() => {
      setTrainingElapsed(getElapsed());
    }, 1000);
    return () => clearInterval(id);
  }, [train.status, train.started_at]);

  const formatDuration = (secs: number) => {
    const m = Math.floor(secs / 60);
    const s = secs % 60;
    return `${m}:${String(s).padStart(2, '0')}`;
  };

  useEffect(() => {
    fetch('/scenes')
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data: {
        scenes: Scene[];
        default_max_steps: number;
        step_presets?: StepPreset[];
        default_train_steps?: number;
      }) => {
        setScenes(data.scenes);
        setDefaultMaxSteps(data.default_max_steps);
        if (data.step_presets && data.step_presets.length > 0) {
          setPresets(data.step_presets);
        }
        if (data.default_train_steps) {
          setTrainSteps(data.default_train_steps);
        }
      })
      .catch((e) => setBootError(e instanceof Error ? e.message : String(e)));
  }, []);

  const executeRun = useCallback(
    async (idx: number, controller: AbortController) => {
      try {
        const res = await fetch(`/scene/${idx}/run`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ max_steps: defaultMaxSteps }),
          signal: controller.signal,
        });
        if (!res.ok) {
          const detail = await res.text();
          throw new Error(`HTTP ${res.status}: ${detail.slice(0, 160)}`);
        }
        const result = (await res.json()) as RunResult & { cancelled?: boolean };
        setRuns((prev) => ({
          ...prev,
          [idx]: result.cancelled
            ? { status: 'cancelled' }
            : { status: 'done', result, finishedAt: Date.now(), cacheBust: Date.now() },
        }));
      } catch (e) {
        if (e instanceof DOMException && e.name === 'AbortError') {
          setRuns((prev) => ({ ...prev, [idx]: { status: 'cancelled' } }));
          return;
        }
        setRuns((prev) => ({
          ...prev,
          [idx]: { status: 'error', message: e instanceof Error ? e.message : String(e) },
        }));
      }
    },
    [defaultMaxSteps],
  );

  const pumpRunAllQueue = useCallback(() => {
    while (
      runAllActiveRef.current < RUN_ALL_CONCURRENCY &&
      runAllQueueRef.current.length > 0
    ) {
      const idx = runAllQueueRef.current.shift()!;
      if (runsRef.current[idx]?.status === 'running') continue;

      runAllActiveRef.current += 1;
      const controller = new AbortController();
      setRuns((prev) => ({
        ...prev,
        [idx]: { status: 'running', startedAt: Date.now(), controller },
      }));
      void executeRun(idx, controller).finally(() => {
        runAllActiveRef.current -= 1;
        pumpRunAllQueue();
      });
    }
  }, [executeRun]);

  const runOne = useCallback(
    (idx: number) => {
      const state = runsRef.current[idx]?.status;
      if (state === 'running') return;
      if (state === 'queued') {
        runAllQueueRef.current = runAllQueueRef.current.filter((i) => i !== idx);
      }
      const controller = new AbortController();
      setRuns((prev) => ({
        ...prev,
        [idx]: { status: 'running', startedAt: Date.now(), controller },
      }));
      void executeRun(idx, controller);
    },
    [executeRun],
  );

  const runAll = useCallback(() => {
    const pending = scenes
      .map((scene) => scene.index)
      .filter((idx) => {
        const state = runsRef.current[idx]?.status;
        return state !== 'running' && state !== 'queued';
      });
    if (pending.length === 0) return;

    setRuns((prev) => {
      const next = { ...prev };
      for (const idx of pending) {
        next[idx] = { status: 'queued' };
      }
      return next;
    });

    runAllQueueRef.current.push(...pending);
    pumpRunAllQueue();
  }, [scenes, pumpRunAllQueue]);

  const startTrain = useCallback(async () => {
    if (train.status === 'running') return;
    killedRef.current = false;
    try {
      const res = await fetch('/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ total_steps: trainSteps }),
      });
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(`HTTP ${res.status}: ${detail.slice(0, 160)}`);
      }
      const data = (await res.json()) as TrainStatus;
      setTrain(data);
    } catch (e) {
      setTrain((prev) => ({
        ...prev,
        status: 'error',
        error: e instanceof Error ? e.message : String(e),
      }));
    }
  }, [train.status, trainSteps]);

  useEffect(() => {
    if (train.status !== 'running') return;
    let cancelled = false;
    const poll = async () => {
      try {
        const res = await fetch('/train/status');
        if (!res.ok) return;
        const data = (await res.json()) as TrainStatus;
        if (!cancelled) setTrain(data);
      } catch {
        // ignore transient errors while polling
      }
    };
    const id = window.setInterval(poll, 2000);
    poll();
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [train.status]);

  useEffect(() => {
    if (train.status === 'done') {
      fetch('/runs').catch(() => {});
    }
  }, [train.status]);

  const killAll = useCallback(async () => {
    if (killing) return;
    killedRef.current = true;
    runAllQueueRef.current = [];
    setKilling(true);
    setRuns((prev) => {
      const next = { ...prev };
      for (const [key, state] of Object.entries(prev)) {
        if (state.status === 'running') {
          try {
            state.controller.abort();
          } catch {
            // noop
          }
          next[Number(key)] = { status: 'cancelled' };
        } else if (state.status === 'queued') {
          next[Number(key)] = { status: 'cancelled' };
        }
      }
      return next;
    });
    try {
      await fetch('/runs/cancel', { method: 'POST' });
    } catch {
      // backend cancel is best-effort; fetch abort already stopped the UI side
    } finally {
      setKilling(false);
    }
  }, [killing]);

  const isTraining = train.status === 'running';
  const anyRunning = isTraining || Object.values(runs).some((r) => r.status === 'running');
  const pendingRunCount = scenes.filter((s) => {
    const status = runs[s.index]?.status;
    return status !== 'running' && status !== 'queued';
  }).length;
  const queuedRunCount = Object.values(runs).filter((r) => r.status === 'queued').length;
  const canRunAll = !isTraining && scenes.length > 0 && pendingRunCount > 0;

  return (
    <div style={{ background: 'var(--bg)', minHeight: '100vh', color: 'var(--ink)' }}>
      <header className="topbar">
        <Link href="/" className="brand">
          <div className="brand-mark" aria-hidden="true"></div>
          <div>
            <div className="brand-name"><em>Battle Angel</em></div>
            <div className="brand-sub mono">EPISODE RUNNER · v0.4.2</div>
          </div>
        </Link>
        <WorkflowNav active="gym" />
        <div className="clock">
          <span className="lbl">UTC</span>
          <span className="val mono">{clockUtc}</span>
          <span className="lbl" style={{ marginLeft: 10 }}>STATUS</span>
          <span className="val mono" style={{ color: anyRunning ? 'var(--red)' : 'var(--ok)' }}>
            {anyRunning ? 'RUNNING' : 'IDLE'}
          </span>
        </div>
      </header>

      <header className="mc-topbar">
        <div>
          <h1 className="mc-title">Training Gym</h1>
          <p className="mc-purpose">
            Train the navigation policy in simulation, then run any gym environment to inspect its animated
            rollout and readiness telemetry before moving into generated scenes.
          </p>
        </div>
        <div className="mc-controls">
          <label className="mc-field">
            <span>Train steps</span>
            <select
              value={presets.some((p) => p.value === trainSteps) ? String(trainSteps) : '__custom__'}
              onChange={(e) => {
                const v = e.target.value;
                if (v === '__custom__') return;
                setTrainSteps(Number(v));
              }}
              disabled={isTraining}
            >
              {presets.map((p) => (
                <option key={p.value} value={p.value}>
                  {p.label}
                </option>
              ))}
              {!presets.some((p) => p.value === trainSteps) && (
                <option value="__custom__">Custom ({trainSteps.toLocaleString()})</option>
              )}
            </select>
          </label>
          <button
            className="btn-primary"
            onClick={startTrain}
            disabled={isTraining}
            style={{ minWidth: 220 }}
            title={`Retrain PPO across ${FIXED_NUM_ENVS} environments`}
          >
            {isTraining
              ? `Training… (${trainSteps.toLocaleString()})`
              : `Train PPO (${trainSteps.toLocaleString()} steps · ${FIXED_NUM_ENVS} envs)`}
          </button>
          <button
            type="button"
            className="btn-secondary"
            onClick={runAll}
            disabled={!canRunAll}
            style={{ minWidth: 120 }}
            title={`Run all ${FIXED_NUM_ENVS} gym environments (${RUN_ALL_CONCURRENCY} at a time)`}
          >
            {queuedRunCount > 0
              ? `Batch running (${queuedRunCount} queued)`
              : pendingRunCount < scenes.length && pendingRunCount > 0
                ? `Run All (${pendingRunCount})`
                : 'Run All'}
          </button>
          <button
            className="btn-danger"
            onClick={killAll}
            disabled={!anyRunning || killing}
            style={{ minWidth: 120 }}
            title="Abort training and all in-flight episodes"
          >
            {killing ? 'Killing…' : 'Kill All'}
          </button>
        </div>
      </header>

      {bootError && (
        <div className="mc-error">
          Could not reach backend: {bootError}. Is FastAPI running on :8000?
        </div>
      )}

      <div className="mc-meta-row">
        <span>Rollout budget <b>{defaultMaxSteps}</b></span>
        <span className="sep">/</span>
        <span>Gym environments <b>{FIXED_NUM_ENVS}</b></span>
        <span className="sep">/</span>
        <span>
          Training <b style={{ color: isTraining ? 'var(--red)' : undefined }}>{train.status}</b>
          {train.total_steps > 0 && <> · <b>{train.total_steps.toLocaleString()}</b> steps</>}
        </span>
        <span className="sep">/</span>
        <span>Running <b>{Object.values(runs).filter((r) => r.status === 'running').length}</b></span>
        <span className="sep">/</span>
        <span>Completed <b>{Object.values(runs).filter((r) => r.status === 'done').length}</b></span>
      </div>

      <section className="mc-grid">
        {scenes.map((scene) => {
          const state: RunState = runs[scene.index] ?? { status: 'idle' };
          const isRunning = state.status === 'running';
          const isQueued = state.status === 'queued';
          const isDone = state.status === 'done';
          const isErr = state.status === 'error';
          const routeStats = isDone ? getRouteStats(scene, state.result) : null;
          return (
            <article
              key={scene.index}
              className={`mc-card${isDone ? ' is-done' : ''}${isErr ? ' is-err' : ''}`}
            >
              <header className="mc-card-hd">
                <div>
                  <div className="mc-card-idx">ENV {String(scene.index + 1).padStart(2, '0')}</div>
                  <h3>{prettyName(scene.name)}</h3>
                </div>
                <span className={`mc-diff mc-diff-${scene.difficulty}`}>{scene.difficulty}</span>
              </header>

              <div className="mc-card-meta">
                <span>start <b>({scene.robot_start.slice(0, 2).map((n) => n.toFixed(1)).join(', ')})</b></span>
                <span>survivor <b>({scene.survivor_pos.slice(0, 2).map((n) => n.toFixed(1)).join(', ')})</b></span>
                {scene.survivor?.buried && (
                  <>
                    <span>status <b>buried</b></span>
                    <span>signal <b>{scene.survivor.signal ?? 'sensor'}</b></span>
                    <span>detect radius <b>{(scene.survivor.detection_radius ?? 0).toFixed(1)}m</b></span>
                  </>
                )}
                <span>obstacles <b>{scene.obstacle_count}</b></span>
                <span>hazards <b>{scene.hazard_count}</b></span>
              </div>

              <div className="mc-card-frame">
                {state.status === 'idle' && (
                  <div className="mc-placeholder">no run yet · press Run</div>
                )}
                {isQueued && <div className="mc-placeholder mc-running">queued…</div>}
                {isRunning && <div className="mc-placeholder mc-running">running episode…</div>}
                {isDone && (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    alt={`${scene.name} episode`}
                    src={`${state.result.gif_url}?t=${state.cacheBust}`}
                  />
                )}
                {isErr && <div className="mc-placeholder mc-err">{state.message}</div>}
                {state.status === 'cancelled' && (
                  <div className="mc-placeholder mc-err">cancelled · press Run</div>
                )}
              </div>

              <footer className="mc-card-ft">
                <div className="mc-stats">
                  {isDone ? (
                    <>
                      <span className={state.result.reached ? 'ok' : 'fail'}>
                        {state.result.reached ? 'REACHED' : 'TIMEOUT'}
                      </span>
                      {state.result.survivor?.buried && (
                        <span className={state.result.detection_event ? 'ok' : 'fail'}>
                          {state.result.detection_event
                            ? `DETECTED step ${state.result.detection_event.step}`
                            : 'NOT DETECTED'}
                        </span>
                      )}
                      <span>steps <b>{state.result.steps}</b>/{state.result.max_steps}</span>
                      <span>reward <b>{state.result.total_reward.toFixed(1)}</b></span>
                      {state.result.detection_event && (
                        <span>
                          ping <b>({state.result.detection_event.robot_pos.slice(0, 2).map((n) => n.toFixed(1)).join(', ')})</b>
                        </span>
                      )}
                    </>
                  ) : (
                    <span className="dim">budget {defaultMaxSteps}</span>
                  )}
                </div>
                <button
                  className="mc-run-btn"
                  onClick={() => runOne(scene.index)}
                  disabled={isRunning || isQueued || isTraining}
                >
                  {isRunning ? '…' : isQueued ? 'queued' : isDone ? 'Re-run' : 'Run'}
                </button>
              </footer>

              {isDone && routeStats && (
                <details className="mc-advanced">
                  <summary>
                    <span>Deployment stats</span>
                    <span className="mc-advanced-peek">
                      efficiency <b>{routeStats.efficiency.toFixed(0)}%</b>
                    </span>
                  </summary>
                  <div className="mc-advanced-panel">
                    <div className="mc-advanced-head">
                      <span>Policy response telemetry</span>
                      <b>{state.result.reached ? 'RESCUE COMPLETE' : 'ROUTE INCOMPLETE'}</b>
                    </div>
                    <div className="mc-kpi-grid">
                      <div><span>Route length</span><b>{routeStats.travelled.toFixed(2)} m</b></div>
                      <div><span>Efficiency</span><b>{routeStats.efficiency.toFixed(1)}%</b></div>
                      <div><span>Final distance</span><b>{routeStats.finalDistance.toFixed(2)} m</b></div>
                      <div><span>Mean reward/step</span><b>{(state.result.total_reward / Math.max(state.result.steps, 1)).toFixed(2)}</b></div>
                      <div><span>Scene load</span><b>{scene.obstacle_count} obs / {scene.hazard_count} haz</b></div>
                      <div>
                        <span>Sensor lock</span>
                        <b>
                          {routeStats.detectionLead == null
                            ? scene.survivor?.buried ? 'none' : 'n/a'
                            : `${routeStats.detectionLead} steps early`}
                        </b>
                      </div>
                    </div>
                    <RouteGraph scene={scene} result={state.result} />
                  </div>
                </details>
              )}
            </article>
          );
        })}
      </section>

      {isTraining && (
        <div className="mc-loading-overlay">
          <div className="mc-loading-card">
            <div className="radar-container">
              <div className="radar-sweep"></div>
              <div className="radar-ring"></div>
              <div className="radar-ring-outer"></div>
              <div className="radar-crosshair-h"></div>
              <div className="radar-crosshair-v"></div>
              <div className="radar-blip"></div>
            </div>

            <div className="training-indicator">
              <span className="dot"></span>
              <span>TRAINING AGENT</span>
            </div>

            <h2 className="mc-loading-title">PPO Optimization In Progress</h2>
            
            <p className="mc-loading-desc">
              Running reinforcement learning rollouts in parallel across {FIXED_NUM_ENVS} simulated environments.
            </p>

            <div className="mc-loading-stats">
              <div className="stat-box">
                <span className="label">Target Steps</span>
                <span className="value mono">{trainSteps.toLocaleString()}</span>
              </div>
              <div className="stat-box">
                <span className="label">Elapsed Time</span>
                <span className="value mono">{formatDuration(trainingElapsed)}</span>
              </div>
              <div className="stat-box">
                <span className="label">Environments</span>
                <span className="value mono">{FIXED_NUM_ENVS} Active</span>
              </div>
            </div>

            <div className="mc-loading-actions">
              <button
                className="btn-danger-large"
                onClick={killAll}
                disabled={killing}
                title="Abort training and cancel all rollout environments"
              >
                {killing ? 'Aborting Training…' : 'Kill All Processes'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
