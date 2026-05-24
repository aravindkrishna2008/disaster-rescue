'use client';

import Link from 'next/link';
import { useEffect, useMemo, useState } from 'react';
import ThreeArena from './ThreeArena';
import WorkflowNav from './WorkflowNav';
import { setActiveSceneId } from './workflowSession';

type Terrain = {
  grid_size: number;
  heights: number[][];
  roughness?: number[][];
  danger?: number[][];
  rigid?: number[][];
};

type Survivor = {
  name: string;
  type: string;
  priority: string;
  pos: number[];
  active: boolean;
};

type EnvScene = {
  name: string;
  description: string;
  difficulty: string;
  robot_start: number[];
  survivor_pos: number[];
  active_survivor: { name: string; type: string; priority: string; pos: number[] };
  survivors: Survivor[];
  obstacles: { asset_id?: string; pos: number[]; size: number[]; color?: string }[];
  hazards: { hazard_id?: string; type?: string; center: number[]; radius: number; color?: string }[];
  terrain?: Terrain | null;
  notes?: string;
};

type Episode = {
  reached: boolean;
  fallen: boolean;
  cancelled?: boolean;
  completion_reason?: string | null;
  steps: number;
  max_steps: number;
  total_reward: number;
  final_dist?: number | null;
  min_dist?: number | null;
  final_heading_error?: number | null;
  obstacle_contacts?: number;
  hazard_steps?: number;
  min_obstacle_clearance?: number | null;
  min_hazard_clearance?: number | null;
  mean_stance_slip?: number;
  mean_assist_force?: number;
  gait_score?: number;
  assist_scale?: number;
  balance_assist_scale?: number;
  frame_count?: number;
  gif_fps?: number;
  gif_duration_seconds?: number;
  wall_time_seconds?: number | null;
  gif_url?: string;
  trajectory?: number[][];
  detection_event?: { step: number; signal?: string; radius?: number; robot_pos?: number[] } | null;
};

type GeneratedSession = {
  scene_id: string;
  default_max_steps: number;
  scene: {
    description: string;
    difficulty: string;
    assets: unknown[];
    hazards: unknown[];
    survivors: { profile: { name: string; type: string; priority: string } }[];
    notes?: string;
  };
  env_scene: EnvScene;
  eval: { score: number; passed: boolean; feedback: string; issues: string[] };
  episode: Episode | null;
  episode_error?: string | null;
};

const formatNum = (value: number | null | undefined, digits = 2) =>
  value == null ? '-' : value.toFixed(digits);

const reasonLabel = (episode: Episode | null) => {
  if (!episode) return 'AWAITING RUN';
  if (episode.cancelled) return 'CANCELLED';
  if (episode.reached) return 'TARGET REACHED';
  if (episode.fallen) return 'ROBOT FALLEN';
  return 'BUDGET EXHAUSTED';
};

function pathDistance(start: number[], trajectory: number[][] | undefined) {
  const points = [start, ...(trajectory ?? [])];
  return points.slice(1).reduce(
    (distance, point, index) =>
      distance + Math.hypot(point[0] - points[index][0], point[1] - points[index][1]),
    0,
  );
}

export default function GeneratedConsole({ sceneId }: { sceneId: string }) {
  const [session, setSession] = useState<GeneratedSession | null>(null);
  const [budget, setBudget] = useState(1500);
  const [status, setStatus] = useState<'loading' | 'ready' | 'running' | 'error'>('loading');
  const [error, setError] = useState('');
  const [gifStamp, setGifStamp] = useState(0);

  useEffect(() => {
    let active = true;
    fetch(`/generated-scenes/${encodeURIComponent(sceneId)}`)
      .then(async (res) => {
        if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail ?? `HTTP ${res.status}`);
        return res.json() as Promise<GeneratedSession>;
      })
      .then((data) => {
        if (!active) return;
        setSession(data);
        setBudget(data.default_max_steps || 1500);
        setActiveSceneId(sceneId);
        setStatus('ready');
      })
      .catch((err: unknown) => {
        if (!active) return;
        setError(err instanceof Error ? err.message : String(err));
        setStatus('error');
      });
    return () => {
      active = false;
    };
  }, [sceneId]);

  async function runEpisode() {
    setStatus('running');
    setError('');
    try {
      const res = await fetch(`/generated-scenes/${encodeURIComponent(sceneId)}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ max_steps: budget }),
      });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail ?? `HTTP ${res.status}`);
      const data = (await res.json()) as GeneratedSession;
      setSession(data);
      setGifStamp(Date.now());
      setStatus('ready');
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
      setStatus('error');
    }
  }

  const episode = session?.episode ?? null;
  const scene = session?.env_scene;
  const display = useMemo(() => {
    if (!scene) return null;
    const target = scene.active_survivor?.pos ?? scene.survivor_pos;
    const type = (scene.active_survivor?.type ?? 'child').toLowerCase();
    const usesChildMarker = type === 'child' || type === 'baby';
    const hidden = { x: 999, y: 999 };
    const points = episode?.trajectory ?? [];
    const final = points[points.length - 1] ?? scene.robot_start;
    const previous = points[points.length - 2] ?? scene.robot_start;
    return {
      targetName: usesChildMarker ? 'CHILD' as const : 'ADULT' as const,
      robotPos: { x: final[0], y: final[1] },
      heading: Math.atan2(final[1] - previous[1], final[0] - previous[0]) * 180 / Math.PI,
      survivors: {
        CHILD: usesChildMarker ? { x: target[0], y: target[1] } : hidden,
        ADULT: usesChildMarker ? hidden : { x: target[0], y: target[1] },
      },
    };
  }, [episode?.trajectory, scene]);

  const travelled = scene && episode ? pathDistance(scene.robot_start, episode.trajectory) : null;
  const direct = scene
    ? Math.hypot(scene.survivor_pos[0] - scene.robot_start[0], scene.survivor_pos[1] - scene.robot_start[1])
    : null;
  const efficiency = travelled && direct ? Math.min(100, direct / travelled * 100) : null;
  const kpis = episode ? [
    ['Completion', reasonLabel(episode)],
    ['Steps', `${episode.steps} / ${episode.max_steps}`],
    ['GIF length', `${formatNum(episode.gif_duration_seconds, 2)} s`],
    ['Frames / FPS', `${episode.frame_count ?? '-'} / ${episode.gif_fps ?? '-'}`],
    ['Reward', formatNum(episode.total_reward, 2)],
    ['Reward / step', formatNum(episode.total_reward / Math.max(episode.steps, 1), 3)],
    ['Final distance', `${formatNum(episode.final_dist, 3)} m`],
    ['Minimum distance', `${formatNum(episode.min_dist, 3)} m`],
    ['Route length', `${formatNum(travelled, 2)} m`],
    ['Route efficiency', `${formatNum(efficiency, 1)}%`],
    ['Obstacle contacts', String(episode.obstacle_contacts ?? '-')],
    ['Hazard steps', String(episode.hazard_steps ?? '-')],
    ['Obstacle clearance', `${formatNum(episode.min_obstacle_clearance, 3)} m`],
    ['Hazard clearance', `${formatNum(episode.min_hazard_clearance, 3)} m`],
    ['Heading error', formatNum(episode.final_heading_error, 3)],
    ['Gait score', formatNum(episode.gait_score, 3)],
    ['Mean stance slip', formatNum(episode.mean_stance_slip, 3)],
    ['Mean assist force', formatNum(episode.mean_assist_force, 3)],
    ['Assist / balance', `${formatNum(episode.assist_scale, 2)} / ${formatNum(episode.balance_assist_scale, 2)}`],
    ['Render wall time', `${formatNum(episode.wall_time_seconds, 3)} s`],
  ] : [];

  return (
    <div className="generated-console">
      <header className="topbar">
        <Link href="/" className="brand">
          <div className="brand-mark" aria-hidden="true" />
          <div>
            <div className="brand-name"><em>Battle Angel</em></div>
            <div className="brand-sub mono">GENERATED SCENE CONSOLE</div>
          </div>
        </Link>
        <WorkflowNav active="console" />
        <div className="clock">
          <span className="lbl">SESSION</span>
          <span className="val mono">{sceneId.slice(-12)}</span>
        </div>
      </header>

      {status === 'loading' && <div className="gc-message mono">Loading generated scene session...</div>}
      {error && <div className="gc-error mono">ERROR / {error}</div>}

      {session && scene && display && (
        <>
          <section className="gc-header">
            <div>
              <div className="generated-eyebrow mono">03 / INTERACTIVE CONSOLE</div>
              <h1>Generated Scene Response</h1>
              <p>{session.scene.description}</p>
              <div className="gc-scene-meta mono">
                <span>{session.scene.difficulty.toUpperCase()}</span>
                <span>{session.scene.assets.length} ASSETS</span>
                <span>{session.scene.hazards.length} HAZARDS</span>
                <span>EVAL {session.eval.score}/100</span>
              </div>
            </div>
            <div className="gc-controls">
              <label className="mc-field">
                <span>Episode budget</span>
                <input
                  type="number"
                  min={100}
                  max={3000}
                  step={100}
                  value={budget}
                  disabled={status === 'running'}
                  onChange={(event) => setBudget(Number(event.target.value))}
                />
              </label>
              <button className="btn-primary" type="button" onClick={runEpisode} disabled={status === 'running'}>
                {status === 'running' ? 'Rendering GIF...' : episode ? 'Re-run Extended GIF' : 'Run Extended GIF'}
              </button>
              <Link className="btn-secondary" href="/generate">New Scene</Link>
            </div>
          </section>

          <div className="gc-statusbar mono">
            <span className={`pill ${episode?.reached ? 'is-ok' : episode ? 'is-fail' : ''}`}>
              <span className="dot" />
              {reasonLabel(episode)}
            </span>
            <span>target <b>{scene.active_survivor.name}</b> / {scene.active_survivor.priority}</span>
            <span>start <b>({scene.robot_start.slice(0, 2).map((v) => v.toFixed(2)).join(', ')})</b></span>
            <span>end object <b>({scene.survivor_pos.slice(0, 2).map((v) => v.toFixed(2)).join(', ')})</b></span>
          </div>

          <main className="gc-grid">
            <section className="gc-visual">
              <header className="gc-panel-hd">
                <h2>Tactical Scene</h2>
                <span className="mono">GEMINI ENVIRONMENT + PPO ROUTE</span>
              </header>
              <div className="gc-arena">
                <ThreeArena
                  robotPos={display.robotPos}
                  robotHeading={Number.isFinite(display.heading) ? display.heading : 0}
                  activeTarget={display.targetName}
                  trajectory={episode?.trajectory ?? null}
                  obstacles={scene.obstacles}
                  hazards={scene.hazards}
                  terrain={scene.terrain}
                  survivors={display.survivors}
                />
              </div>
            </section>

            <section className="gc-gif">
              <header className="gc-panel-hd">
                <h2>Rendered Rollout</h2>
                <span className="mono">{episode ? `${formatNum(episode.gif_duration_seconds, 2)} SEC` : 'NOT RUN'}</span>
              </header>
              <div className="gc-gif-frame">
                {episode?.gif_url ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={`${episode.gif_url}?t=${gifStamp}`} alt="Extended generated-scene PPO rollout" />
                ) : (
                  <p>Run the policy to render the generated-scene GIF.</p>
                )}
              </div>
            </section>

            <section className="gc-telemetry">
              <header className="gc-panel-hd">
                <h2>Episode Telemetry</h2>
                <span className="mono">{episode?.completion_reason ?? 'AWAITING POLICY'}</span>
              </header>
              {episode ? (
                <dl className="gc-kpis mono">
                  {kpis.map(([label, value]) => (
                    <div key={label}>
                      <dt>{label}</dt>
                      <dd>{value}</dd>
                    </div>
                  ))}
                </dl>
              ) : (
                <p className="gc-empty">No rollout has been rendered yet.</p>
              )}
            </section>

            <section className="gc-detail">
              <header className="gc-panel-hd">
                <h2>Scene Data</h2>
                <span className="mono">EVALAGENT {session.eval.passed ? 'PASSED' : 'REJECTED'}</span>
              </header>
              <dl className="gc-data mono">
                <div><dt>Eval feedback</dt><dd>{session.eval.feedback}</dd></div>
                <div><dt>Survivor</dt><dd>{scene.active_survivor.name} / {scene.active_survivor.type} / {scene.active_survivor.priority}</dd></div>
                <div><dt>Geometry</dt><dd>{scene.obstacles.length} obstacles / {scene.hazards.length} hazards</dd></div>
                <div><dt>Target detection</dt><dd>{episode?.detection_event ? `step ${episode.detection_event.step}` : 'none recorded'}</dd></div>
                <div><dt>Notes</dt><dd>{session.scene.notes || '-'}</dd></div>
              </dl>
              <details className="gc-raw">
                <summary>Raw generated environment JSON</summary>
                <pre>{JSON.stringify(scene, null, 2)}</pre>
              </details>
            </section>
          </main>
        </>
      )}
    </div>
  );
}
