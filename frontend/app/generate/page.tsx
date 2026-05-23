'use client';

import Link from 'next/link';
import { useState } from 'react';

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
  gif_path?: string;
  detection_event?: { step: number; signal?: string } | null;
  cancelled?: boolean;
};

type SceneResult = {
  scene: {
    description: string;
    difficulty: string;
    survivors: { profile: { name: string; type: string; priority: string }; position: number[] }[];
    assets: { asset_id: string; position: number[] }[];
    hazards: { hazard_id: string; type: string; severity: string }[];
    notes: string;
  };
  eval: EvalResult;
  episode: EpisodeResult | null;
  episode_error?: string;
};

const DIFFICULTIES = ['easy', 'medium', 'hard'] as const;

export default function GeneratePage() {
  const [description, setDescription] = useState('');
  const [difficulty, setDifficulty] = useState<'easy' | 'medium' | 'hard'>('medium');
  const [survivorCount, setSurvivorCount] = useState(2);
  const [skipEpisode, setSkipEpisode] = useState(false);
  const [theme, setTheme] = useState('');
  const [promptLoading, setPromptLoading] = useState(false);
  const [status, setStatus] = useState<'idle' | 'loading' | 'done' | 'error'>('idle');
  const [result, setResult] = useState<SceneResult | null>(null);
  const [errorMsg, setErrorMsg] = useState('');

  async function handleGeneratePrompt() {
    setPromptLoading(true);
    try {
      const res = await fetch('/generate-prompt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ difficulty, survivor_count: survivorCount, theme }),
      });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
      const data = await res.json();
      setDescription(data.prompt);
    } catch (e) {
      setErrorMsg(e instanceof Error ? e.message : String(e));
    } finally {
      setPromptLoading(false);
    }
  }

  async function handleGenerate() {
    if (!description.trim()) return;
    setStatus('loading');
    setResult(null);
    setErrorMsg('');

    try {
      const res = await fetch('/generate-scene', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          description: description.trim(),
          difficulty,
          survivor_count: survivorCount,
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
    } catch (e) {
      setErrorMsg(e instanceof Error ? e.message : String(e));
      setStatus('error');
    }
  }

  const scoreColor = (score: number) =>
    score >= 80 ? 'var(--ok)' : score >= 60 ? '#f59e0b' : 'var(--red)';

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
        <div className="nav-tabs">
          <Link href="/" className="nav-tab">Overview</Link>
          <Link href="/console" className="nav-tab">Interactive Console</Link>
          <Link href="/mission-control" className="nav-tab">Mission Control</Link>
          <Link href="/generate" className="nav-tab is-active">Scene Generator</Link>
        </div>
      </header>

      <div style={{ maxWidth: 860, margin: '0 auto', padding: '40px 24px' }}>
        <div style={{ marginBottom: 32 }}>
          <h1 style={{ fontSize: 24, fontWeight: 700, letterSpacing: 1, textTransform: 'uppercase', marginBottom: 6 }}>
            Scene Generator
          </h1>
          <p style={{ color: 'var(--ink-dim)', fontSize: 13 }}>
            Describe a disaster scenario. ScenarioAgent generates the scene. EvalAgent scores it.
            If it passes, the robot runs it.
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
              {promptLoading ? 'Asking Gemini…' : '✦ Generate with Gemini'}
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
              placeholder="e.g. Collapsed apartment building after earthquake, two survivors trapped in rubble"
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

            <div style={{ flex: '1 1 140px' }}>
              <label style={{ display: 'block', fontSize: 11, fontFamily: 'monospace', color: 'var(--ink-dim)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: 1 }}>
                Survivors
              </label>
              <input
                type="number"
                min={1}
                max={6}
                value={survivorCount}
                onChange={e => setSurvivorCount(Math.max(1, Math.min(6, Number(e.target.value))))}
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

            {/* Episode result */}
            {result.episode && (
              <div style={{ background: 'var(--surface)', border: `1px solid ${result.episode.reached ? 'var(--ok)' : 'var(--rule)'}`, borderRadius: 6, padding: 20 }}>
                <div style={{ fontSize: 11, fontFamily: 'monospace', color: 'var(--ink-dim)', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 12 }}>
                  Robot Episode
                </div>
                <div style={{ display: 'flex', gap: 28, fontFamily: 'monospace', fontSize: 13, flexWrap: 'wrap' }}>
                  <span>
                    reached: <strong style={{ color: result.episode.reached ? 'var(--ok)' : 'var(--red)' }}>
                      {result.episode.reached ? 'YES' : 'NO'}
                    </strong>
                  </span>
                  <span>steps: <strong style={{ color: 'var(--ink)' }}>{result.episode.steps}</strong></span>
                  <span>reward: <strong style={{ color: 'var(--ink)' }}>{result.episode.total_reward.toFixed(2)}</strong></span>
                </div>
              </div>
            )}

            {result.episode_error && (
              <div style={{ fontFamily: 'monospace', fontSize: 12, color: 'var(--red)', padding: '8px 0' }}>
                Episode error: {result.episode_error}
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
