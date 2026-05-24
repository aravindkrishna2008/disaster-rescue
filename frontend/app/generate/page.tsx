'use client';

import Link from 'next/link';
import { useState } from 'react';
import ThreeArena from '../ThreeArena';
import WorkflowNav from '../WorkflowNav';

type EvalResult = {
  score: number;
  passed: boolean;
  issues: string[];
  feedback: string;
};

type EpisodeResult = {
  reached: boolean;
  steps: number;
  total_reward: number;
  final_dist?: number | null;
  gif_url?: string;
  gif_path?: string;
  trajectory?: number[][];
  min_dist?: number | null;
  obstacle_contacts?: number;
  hazard_steps?: number;
  detection_event?: { step: number; signal?: string } | null;
  cancelled?: boolean;
};

type EnvScene = {
  robot_start: number[];
  survivor_pos: number[];
  active_survivor?: { name: string; type: string; priority: string; pos: number[] };
  survivors?: { name: string; type: string; priority: string; pos: number[]; active: boolean }[];
  obstacles: { asset_id?: string; pos: number[]; size: number[]; color?: string }[];
  hazards: { hazard_id?: string; type?: string; center: number[]; radius: number; color?: string }[];
  terrain?: {
    grid_size: number;
    heights: number[][];
    roughness?: number[][];
    danger?: number[][];
    rigid?: number[][];
  };
};

type SceneResult = {
  scene_id: string;
  default_max_steps: number;
  scene: {
    description: string;
    difficulty: string;
    survivors: { profile: { name: string; type: string; priority: string }; position: number[] }[];
    assets: { asset_id: string; position: number[] }[];
    hazards: { hazard_id: string; type: string; severity: string }[];
    notes: string;
  };
  env_scene: EnvScene;
  eval: EvalResult;
  episode: EpisodeResult | null;
  episode_error?: string;
};

const DIFFICULTIES = ['easy', 'medium', 'hard'] as const;
const SURVIVOR_COUNT = 1;

export default function GeneratePage() {
  const [description, setDescription] = useState('');
  const [difficulty, setDifficulty] = useState<'easy' | 'medium' | 'hard'>('medium');
  const [skipEpisode, setSkipEpisode] = useState(false);
  const [theme, setTheme] = useState('');
  const [promptLoading, setPromptLoading] = useState(false);
  const [status, setStatus] = useState<'idle' | 'loading' | 'done' | 'error'>('idle');
  const [result, setResult] = useState<SceneResult | null>(null);
  const [errorMsg, setErrorMsg] = useState('');

  async function runSceneGeneration(sceneDescription: string) {
    const trimmedDescription = sceneDescription.trim();
    if (!trimmedDescription) return;

    setStatus('loading');
    setResult(null);
    setErrorMsg('');

    const res = await fetch('/generate-scene', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        description: trimmedDescription,
        difficulty,
        survivor_count: SURVIVOR_COUNT,
        theme: theme.trim() || null,
        skip_episode: skipEpisode,
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || res.statusText);
    }
    const data: SceneResult = await res.json();
    setResult(data);
    setStatus('done');
  }

  async function handleGeneratePrompt() {
    setPromptLoading(true);
    setErrorMsg('');
    try {
      const res = await fetch('/generate-prompt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ difficulty, survivor_count: SURVIVOR_COUNT, theme }),
      });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
      const data = await res.json();
      setDescription(data.prompt);
      await runSceneGeneration(data.prompt);
    } catch (e) {
      setStatus('error');
      setErrorMsg(e instanceof Error ? e.message : String(e));
    } finally {
      setPromptLoading(false);
    }
  }

  async function handleGenerate() {
    try {
      await runSceneGeneration(description);
    } catch (e) {
      setErrorMsg(e instanceof Error ? e.message : String(e));
      setStatus('error');
    }
  }

  const scoreColor = (score: number) =>
    score >= 80 ? 'var(--ok)' : score >= 60 ? '#f59e0b' : 'var(--red)';
  const episodeGif = result?.episode?.gif_url
    ?? (result?.episode?.gif_path
      ? `/gifs/${result.episode.gif_path.split('/').pop()}`
      : null);

  const visualization = result
    ? (() => {
        const active = result.env_scene.active_survivor?.pos ?? result.env_scene.survivor_pos;
        const survivorType = (result.env_scene.active_survivor?.type ?? 'child').toLowerCase();
        const isChild = survivorType === 'child' || survivorType === 'baby';
        const trajectory = result.episode?.trajectory ?? null;
        const finalPoint = trajectory?.[trajectory.length - 1] ?? result.env_scene.robot_start;
        const previousPoint = trajectory && trajectory.length > 1
          ? trajectory[trajectory.length - 2]
          : result.env_scene.robot_start;
        const heading = Math.atan2(finalPoint[1] - previousPoint[1], finalPoint[0] - previousPoint[0]) * 180 / Math.PI;
        const offMap = { x: 999, y: 999 };
        const activePos = { x: active[0], y: active[1] };
        return {
          targetName: isChild ? 'CHILD' as const : 'ADULT' as const,
          robotPos: { x: finalPoint[0], y: finalPoint[1] },
          heading: Number.isFinite(heading) ? heading : 0,
          trajectory,
          survivors: {
            CHILD: isChild ? activePos : offMap,
            ADULT: isChild ? offMap : activePos,
          },
        };
      })()
    : null;

  return (
    <div style={{ background: 'var(--bg)', minHeight: '100vh', color: 'var(--ink)' }}>
      <header className="topbar">
        <Link href="/" className="brand">
          <div className="brand-mark" aria-hidden="true" />
          <div>
            <div className="brand-name"><em>Battle Angel</em></div>
            <div className="brand-sub mono">SCENE GENERATOR · v0.1.0</div>
          </div>
        </Link>
        <WorkflowNav active="generate" />
      </header>

      <div style={{ maxWidth: 860, margin: '0 auto', padding: '40px 24px' }}>
        <div style={{ marginBottom: 32 }}>
          <h1 style={{ fontSize: 24, fontWeight: 700, letterSpacing: 1, textTransform: 'uppercase', marginBottom: 6 }}>
            02 / Scene Generator
          </h1>
          <p style={{ color: 'var(--ink-dim)', fontSize: 13 }}>
            After the Training Gym, describe a deployment candidate. ScenarioAgent builds the scene and
            EvalAgent scores it. Passing scenes run the PPO policy for playback and readiness statistics.
          </p>
        </div>

        {/* Form */}
        <div style={{ background: 'var(--surface)', border: '1px solid var(--rule)', borderRadius: 6, padding: 24, marginBottom: 28 }}>
          {/* AI prompt generator */}
          <div style={{ display: 'flex', gap: 10, marginBottom: 14, alignItems: 'flex-end', flexWrap: 'wrap' }}>
            <div style={{ flex: '1 1 200px' }}>
              <label style={{ display: 'block', fontSize: 11, fontFamily: 'monospace', color: 'var(--ink-dim)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: 1 }}>
                Theme hint (optional)
              </label>
              <input
                type="text"
                value={theme}
                onChange={e => setTheme(e.target.value)}
                placeholder="e.g. flood, earthquake, chemical plant"
                style={{
                  width: '100%',
                  background: 'var(--bg)',
                  border: '1px solid var(--rule)',
                  borderRadius: 4,
                  color: 'var(--ink)',
                  fontFamily: 'monospace',
                  fontSize: 13,
                  padding: '8px 10px',
                  boxSizing: 'border-box',
                }}
              />
            </div>
            <button
              onClick={handleGeneratePrompt}
              disabled={promptLoading}
              className="btn-secondary"
              style={{ opacity: promptLoading ? 0.5 : 1, whiteSpace: 'nowrap' }}
            >
              {promptLoading ? 'Asking Gemini…' : 'Generate with Gemini'}
            </button>
          </div>

          <div style={{ marginBottom: 18 }}>
            <label style={{ display: 'block', fontSize: 11, fontFamily: 'monospace', color: 'var(--ink-dim)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: 1 }}>
              Scenario Description
            </label>
            <textarea
              value={description}
              onChange={e => setDescription(e.target.value)}
              rows={3}
              placeholder="e.g. Collapsed apartment building after earthquake, one survivor trapped in rubble"
              style={{
                width: '100%',
                background: 'var(--bg)',
                border: '1px solid var(--rule)',
                borderRadius: 4,
                color: 'var(--ink)',
                fontFamily: 'monospace',
                fontSize: 13,
                padding: '10px 12px',
                resize: 'vertical',
                boxSizing: 'border-box',
              }}
            />
          </div>

          <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 18 }}>
            <div style={{ flex: '1 1 140px' }}>
              <label style={{ display: 'block', fontSize: 11, fontFamily: 'monospace', color: 'var(--ink-dim)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: 1 }}>
                Difficulty
              </label>
              <select
                value={difficulty}
                onChange={e => setDifficulty(e.target.value as typeof difficulty)}
                style={{
                  width: '100%',
                  background: 'var(--bg)',
                  border: '1px solid var(--rule)',
                  borderRadius: 4,
                  color: 'var(--ink)',
                  fontFamily: 'monospace',
                  fontSize: 13,
                  padding: '8px 10px',
                }}
              >
                {DIFFICULTIES.map(d => (
                  <option key={d} value={d}>{d.toUpperCase()}</option>
                ))}
              </select>
            </div>

            <div style={{ flex: '1 1 200px', display: 'flex', alignItems: 'flex-end', paddingBottom: 2 }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', fontSize: 13, fontFamily: 'monospace', color: 'var(--ink-dim)' }}>
                <input
                  type="checkbox"
                  checked={skipEpisode}
                  onChange={e => setSkipEpisode(e.target.checked)}
                />
                Skip robot episode
              </label>
            </div>
          </div>

          <button
            onClick={handleGenerate}
            disabled={status === 'loading' || !description.trim()}
            className="btn-primary"
            style={{ opacity: status === 'loading' || !description.trim() ? 0.5 : 1 }}
          >
            {status === 'loading' ? 'Generating…' : 'Generate Scene'}
          </button>
        </div>

        {/* Loading */}
        {status === 'loading' && (
          <div style={{ fontFamily: 'monospace', fontSize: 13, color: 'var(--ink-dim)', padding: '16px 0' }}>
            <span style={{ color: 'var(--ok)' }}>▶</span> ScenarioAgent generating scene…<br />
            <span style={{ color: 'var(--ink-dim)', marginLeft: 16 }}>EvalAgent critique pending</span>
          </div>
        )}

        {/* Error */}
        {status === 'error' && (
          <div style={{ background: 'var(--surface)', border: '1px solid var(--red)', borderRadius: 6, padding: 16, fontFamily: 'monospace', fontSize: 13, color: 'var(--red)' }}>
            ERROR · {errorMsg}
          </div>
        )}

        {/* Result */}
        {status === 'done' && result && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {visualization && (
              <div style={{ background: 'var(--surface)', border: '1px solid var(--rule)', borderRadius: 6, overflow: 'hidden' }}>
                <div style={{ height: 460 }}>
                  <ThreeArena
                    robotPos={visualization.robotPos}
                    robotHeading={visualization.heading}
                    activeTarget={visualization.targetName}
                    trajectory={visualization.trajectory}
                    obstacles={result.env_scene.obstacles}
                    hazards={result.env_scene.hazards}
                    terrain={result.env_scene.terrain ?? null}
                    survivors={visualization.survivors}
                  />
                </div>
              </div>
            )}

            {/* Eval card */}
            <div style={{ background: 'var(--surface)', border: `1px solid ${result.eval.passed ? 'var(--ok)' : 'var(--red)'}`, borderRadius: 6, padding: 20 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 12 }}>
                <span style={{ fontFamily: 'monospace', fontSize: 11, color: 'var(--ink-dim)', textTransform: 'uppercase', letterSpacing: 1 }}>
                  EvalAgent
                </span>
                <span style={{ fontFamily: 'monospace', fontSize: 28, fontWeight: 700, color: scoreColor(result.eval.score) }}>
                  {result.eval.score}
                </span>
                <span style={{ fontFamily: 'monospace', fontSize: 13, color: result.eval.passed ? 'var(--ok)' : 'var(--red)', fontWeight: 700 }}>
                  {result.eval.passed ? '✓ PASSED' : '✗ REJECTED'}
                </span>
              </div>
              <p style={{ fontSize: 13, margin: '0 0 10px', color: 'var(--ink)' }}>{result.eval.feedback}</p>
              {result.eval.issues.length > 0 && (
                <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12, color: 'var(--ink-dim)', fontFamily: 'monospace' }}>
                  {result.eval.issues.map((issue, i) => <li key={i}>{issue}</li>)}
                </ul>
              )}
            </div>

            {/* Scene summary */}
            <div style={{ background: 'var(--surface)', border: '1px solid var(--rule)', borderRadius: 6, padding: 20 }}>
              <div style={{ fontSize: 11, fontFamily: 'monospace', color: 'var(--ink-dim)', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 12 }}>
                Generated Scene
              </div>
              <p style={{ margin: '0 0 12px', fontSize: 14 }}>{result.scene.description}</p>
              <div style={{ display: 'flex', gap: 24, fontFamily: 'monospace', fontSize: 12, color: 'var(--ink-dim)', flexWrap: 'wrap' }}>
                <span>difficulty: <strong style={{ color: 'var(--ink)' }}>{result.scene.difficulty}</strong></span>
                <span>assets: <strong style={{ color: 'var(--ink)' }}>{result.scene.assets.length}</strong></span>
                <span>hazards: <strong style={{ color: 'var(--ink)' }}>{result.scene.hazards.length}</strong></span>
                <span>survivors: <strong style={{ color: 'var(--ink)' }}>{result.scene.survivors.length}</strong></span>
              </div>
              {result.scene.survivors.length > 0 && (
                <div style={{ marginTop: 12, fontSize: 12, fontFamily: 'monospace', color: 'var(--ink-dim)' }}>
                  {result.scene.survivors.map((s, i) => (
                    <div key={i}>
                      {s.profile.name} · {s.profile.type} · <span style={{ color: s.profile.priority === 'critical' ? 'var(--red)' : s.profile.priority === 'high' ? '#f59e0b' : 'var(--ink-dim)' }}>{s.profile.priority}</span>
                    </div>
                  ))}
                </div>
              )}
              {result.scene.notes && (
                <p style={{ marginTop: 10, fontSize: 12, fontStyle: 'italic', color: 'var(--ink-dim)' }}>{result.scene.notes}</p>
              )}
            </div>

            {/* Episode playback and deployment stats */}
            {result.episode && (
              <div className="generated-playback">
                <div className="generated-playback-hd">
                  <div>
                    <span className="generated-eyebrow mono">03 / POLICY PLAYBACK</span>
                    <h2>Generated Scene Response</h2>
                    <p>Animated rollout and initial deployment-readiness statistics for this scene.</p>
                  </div>
                  <Link href={`/console?scene_id=${encodeURIComponent(result.scene_id)}`} className="btn-secondary">
                    Open Interactive Console
                  </Link>
                </div>
                <div className="generated-playback-body">
                  <div className="generated-animation">
                    {episodeGif ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img src={`${episodeGif}?t=${Date.now()}`} alt="PPO rollout in generated scene" />
                    ) : (
                      <div className="generated-animation-empty">
                        Animation not available for this episode.
                      </div>
                    )}
                  </div>
                  <dl className="generated-kpis mono">
                    <div>
                      <dt>Target reached</dt>
                      <dd className={result.episode.reached ? 'ok' : 'fail'}>
                        {result.episode.reached ? 'YES' : 'NO'}
                      </dd>
                    </div>
                    <div><dt>Steps</dt><dd>{result.episode.steps}</dd></div>
                    <div><dt>Total reward</dt><dd>{result.episode.total_reward.toFixed(2)}</dd></div>
                    <div>
                      <dt>Final distance</dt>
                      <dd>{result.episode.final_dist == null ? '—' : `${result.episode.final_dist.toFixed(2)} m`}</dd>
                    </div>
                    <div><dt>Obstacle contacts</dt><dd>{result.episode.obstacle_contacts ?? '—'}</dd></div>
                    <div><dt>Hazard steps</dt><dd>{result.episode.hazard_steps ?? '—'}</dd></div>
                  </dl>
                </div>
              </div>
            )}

            {result.episode_error && (
              <div style={{ fontFamily: 'monospace', fontSize: 12, color: 'var(--red)', padding: '8px 0' }}>
                Episode error: {result.episode_error}
              </div>
            )}

            {result.eval.passed && !result.episode && (
              <div className="generated-console-invite">
                <div>
                  <span className="generated-eyebrow mono">03 / POLICY PLAYBACK</span>
                  <strong>Scene is ready for an interactive extended run.</strong>
                </div>
                <Link href={`/console?scene_id=${encodeURIComponent(result.scene_id)}`} className="btn-secondary">
                  Open Interactive Console
                </Link>
              </div>
            )}

            {!result.eval.passed && (
              <div style={{ fontFamily: 'monospace', fontSize: 12, color: 'var(--ink-dim)', padding: '4px 0' }}>
                Scene rejected by EvalAgent (score &lt; 60). Robot episode skipped.
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
