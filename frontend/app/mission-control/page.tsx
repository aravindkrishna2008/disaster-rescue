'use client';

import Link from 'next/link';
import { useCallback, useEffect, useState } from 'react';

type Scene = {
  index: number;
  name: string;
  difficulty: string;
  robot_start: number[];
  survivor_pos: number[];
  survivor?: SurvivorMeta;
  obstacle_count: number;
  hazard_count: number;
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
};

type RunState =
  | { status: 'idle' }
  | { status: 'running'; startedAt: number }
  | { status: 'done'; result: RunResult; finishedAt: number; cacheBust: number }
  | { status: 'error'; message: string };

const prettyName = (n: string) =>
  n.split('_').map((w) => w[0].toUpperCase() + w.slice(1)).join(' ');

export default function MissionControlPage() {
  const [scenes, setScenes] = useState<Scene[]>([]);
  const [runs, setRuns] = useState<Record<number, RunState>>({});
  const [maxSteps, setMaxSteps] = useState(300);
  const [numEnvs, setNumEnvs] = useState(6);
  const [defaultMaxSteps, setDefaultMaxSteps] = useState(300);
  const [bootError, setBootError] = useState<string | null>(null);
  const [runningAll, setRunningAll] = useState(false);

  useEffect(() => {
    fetch('/scenes')
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data: { scenes: Scene[]; default_max_steps: number }) => {
        setScenes(data.scenes);
        setDefaultMaxSteps(data.default_max_steps);
        setMaxSteps(data.default_max_steps);
        setNumEnvs(data.scenes.length);
      })
      .catch((e) => setBootError(e instanceof Error ? e.message : String(e)));
  }, []);

  const runOne = useCallback(
    async (idx: number) => {
      setRuns((prev) => ({ ...prev, [idx]: { status: 'running', startedAt: Date.now() } }));
      try {
        const res = await fetch(`/scene/${idx}/run`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ max_steps: maxSteps }),
        });
        if (!res.ok) {
          const detail = await res.text();
          throw new Error(`HTTP ${res.status}: ${detail.slice(0, 160)}`);
        }
        const result = (await res.json()) as RunResult;
        setRuns((prev) => ({
          ...prev,
          [idx]: { status: 'done', result, finishedAt: Date.now(), cacheBust: Date.now() },
        }));
      } catch (e) {
        setRuns((prev) => ({
          ...prev,
          [idx]: { status: 'error', message: e instanceof Error ? e.message : String(e) },
        }));
      }
    },
    [maxSteps],
  );

  const runAll = useCallback(async () => {
    if (runningAll) return;
    setRunningAll(true);
    const indices = scenes.slice(0, numEnvs).map((s) => s.index);
    for (const idx of indices) {
      await runOne(idx);
    }
    setRunningAll(false);
  }, [scenes, numEnvs, runOne, runningAll]);

  const anyRunning = runningAll || Object.values(runs).some((r) => r.status === 'running');

  return (
    <div style={{ background: 'var(--bg)', minHeight: '100vh', color: 'var(--ink)' }}>
      <header className="mc-topbar">
        <div>
          <Link href="/" className="mc-back">← Back to overview</Link>
          <h1 className="mc-title">
            Mission Control <span>— Multi-Environment Rollout Suite</span>
          </h1>
        </div>
        <div className="mc-controls">
          <label className="mc-field">
            <span>Max steps</span>
            <input
              type="number"
              min={50}
              max={1200}
              step={25}
              value={maxSteps}
              onChange={(e) => setMaxSteps(Math.max(1, Number(e.target.value) || 0))}
              disabled={anyRunning}
            />
          </label>
          <label className="mc-field">
            <span>Environments</span>
            <input
              type="number"
              min={1}
              max={scenes.length || 6}
              step={1}
              value={numEnvs}
              onChange={(e) =>
                setNumEnvs(Math.min(scenes.length || 6, Math.max(1, Number(e.target.value) || 1)))
              }
              disabled={anyRunning}
            />
          </label>
          <button
            className="btn-primary"
            onClick={runAll}
            disabled={anyRunning || scenes.length === 0}
            style={{ minWidth: 180 }}
          >
            {runningAll ? 'Running…' : `Run ${numEnvs} Episode${numEnvs === 1 ? '' : 's'}`}
          </button>
        </div>
      </header>

      {bootError && (
        <div className="mc-error">
          Could not reach backend: {bootError}. Is FastAPI running on :8000?
        </div>
      )}

      <div className="mc-meta-row">
        <span>Default budget <b>{defaultMaxSteps}</b></span>
        <span className="sep">/</span>
        <span>Available scenes <b>{scenes.length}</b></span>
        <span className="sep">/</span>
        <span>Running <b>{Object.values(runs).filter((r) => r.status === 'running').length}</b></span>
        <span className="sep">/</span>
        <span>Completed <b>{Object.values(runs).filter((r) => r.status === 'done').length}</b></span>
      </div>

      <section className="mc-grid">
        {scenes.map((scene) => {
          const state: RunState = runs[scene.index] ?? { status: 'idle' };
          const isRunning = state.status === 'running';
          const isDone = state.status === 'done';
          const isErr = state.status === 'error';
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
                {isRunning && <div className="mc-placeholder mc-running">running episode…</div>}
                {isDone && (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    alt={`${scene.name} episode`}
                    src={`${state.result.gif_url}?t=${state.cacheBust}`}
                  />
                )}
                {isErr && <div className="mc-placeholder mc-err">{state.message}</div>}
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
                    <span className="dim">budget {maxSteps}</span>
                  )}
                </div>
                <button
                  className="mc-run-btn"
                  onClick={() => runOne(scene.index)}
                  disabled={isRunning || runningAll}
                >
                  {isRunning ? '…' : isDone ? 'Re-run' : 'Run'}
                </button>
              </footer>
            </article>
          );
        })}
      </section>
    </div>
  );
}
