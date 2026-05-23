'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import {
  BENCHMARK_SCENES,
  MDP_CONSTANTS,
  OBSERVATION_SPEC,
  RESEARCH_TASKS,
  REWARD_TERMS,
} from './landingReportData';

type EvalScene = {
  scene_index: number;
  scene_name: string;
  reached: boolean;
  steps: number;
  total_reward: number;
  detection_event: boolean;
  gif?: string;
};

type EvalPayload = {
  model?: string;
  scene_count?: number;
  success_count?: number;
  detection_count?: number;
  mean_reward?: number;
  max_steps?: number;
  scenes?: EvalScene[];
};

type RunMeta = {
  id: string;
  name: string;
  exportedAt?: string;
  git_commit?: string;
};

const STATUS_LABEL: Record<string, string> = {
  complete: 'DONE',
  in_progress: 'ACTIVE',
  blocked: 'BLOCKED',
  planned: 'PLANNED',
};

function formatSceneName(name: string) {
  return name.replace(/_/g, ' ');
}

function gifUrl(runId: string, gifPath?: string) {
  if (!gifPath) return null;
  return `/run-gifs/${runId}/${gifPath}`;
}

export default function LandingReport() {
  const [runMeta, setRunMeta] = useState<RunMeta | null>(null);
  const [evalData, setEvalData] = useState<EvalPayload | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const listRes = await fetch('/runs');
        if (!listRes.ok) throw new Error('runs list failed');
        const { runs } = (await listRes.json()) as { runs: { id: string; name: string }[] };
        const primary = runs?.find((r) => r.id.includes('buried_detection')) ?? runs?.[0];
        if (!primary) throw new Error('no runs');

        const manifestRes = await fetch(`/runs/${primary.id}`);
        if (!manifestRes.ok) throw new Error('manifest failed');
        const manifest = (await manifestRes.json()) as Record<string, unknown>;

        if (!cancelled) {
          setRunMeta({
            id: primary.id,
            name: String(manifest.name ?? primary.name),
            exportedAt: manifest.exportedAt as string | undefined,
            git_commit: manifest.git_commit as string | undefined,
          });
          setEvalData(manifest.eval as EvalPayload | undefined ?? null);
        }
      } catch {
        if (!cancelled) {
          setRunMeta(null);
          setEvalData(null);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const evalByName = new Map(
    (evalData?.scenes ?? []).map((s) => [s.scene_name, s]),
  );

  return (
    <>
      {/* Section 03 — Report header & methodology */}
      <section className="landing-section report-section">
        <div className="landing-section-hd">
          <h2>
            Technical Report <span>— Benchmark & Methodology</span>
          </h2>
          <span className="sec-idx mono">[ SECTION 03 / REPORT ]</span>
        </div>

        <div className="report-doc-header">
          <dl className="report-meta-grid">
            <div>
              <dt>Document</dt>
              <dd>BA-RESCUE-EVAL-v0.4</dd>
            </div>
            <div>
              <dt>Primary model</dt>
              <dd>{runMeta?.name ?? '—'}</dd>
            </div>
            <div>
              <dt>Git commit</dt>
              <dd className="mono">{runMeta?.git_commit ?? '—'}</dd>
            </div>
            <div>
              <dt>Last export (UTC)</dt>
              <dd className="mono">
                {runMeta?.exportedAt
                  ? new Date(runMeta.exportedAt).toISOString().slice(0, 19).replace('T', ' ')
                  : '—'}
              </dd>
            </div>
          </dl>
        </div>

        <p className="report-abstract">
          We evaluate a grounded 2D navigation policy in eight fixed disaster scenes rendered with MuJoCo.
          The agent receives a 21-dimensional state vector encoding pose, nearest obstacles, procedural terrain
          effects, and buried-survivor detection signals. Success is defined as reaching within{' '}
          <strong>{MDP_CONSTANTS.reachTolerance}</strong> within the eval step budget (
          <strong>{MDP_CONSTANTS.maxStepsEval}</strong> steps). Training uses vectorized PPO across all eight
          scenes with shared weights.
        </p>

        <div className="report-kpi-row">
          <div className="report-kpi">
            <span className="report-kpi-label">Scenes</span>
            <span className="report-kpi-value">{BENCHMARK_SCENES.length}</span>
          </div>
          <div className="report-kpi">
            <span className="report-kpi-label">Obs dim</span>
            <span className="report-kpi-value">{MDP_CONSTANTS.obsDim}D</span>
          </div>
          <div className="report-kpi">
            <span className="report-kpi-label">Reach rate</span>
            <span className="report-kpi-value">
              {evalData
                ? `${evalData.success_count ?? 0}/${evalData.scene_count ?? 8}`
                : loading
                  ? '…'
                  : '—'}
            </span>
          </div>
          <div className="report-kpi">
            <span className="report-kpi-label">Detections</span>
            <span className="report-kpi-value">
              {evalData?.detection_count != null ? evalData.detection_count : loading ? '…' : '—'}
            </span>
          </div>
          <div className="report-kpi">
            <span className="report-kpi-label">Mean reward</span>
            <span className="report-kpi-value">
              {evalData?.mean_reward != null
                ? evalData.mean_reward.toFixed(1)
                : loading
                  ? '…'
                  : '—'}
            </span>
          </div>
        </div>
      </section>

      {/* Section 04 — Task benchmark suite */}
      <section className="landing-section report-section">
        <div className="landing-section-hd">
          <h2>
            Benchmark Suite <span>— Task Definitions</span>
          </h2>
          <span className="sec-idx mono">[ SECTION 04 / TASKS ]</span>
        </div>

        <p className="report-lede">
          Table I lists the eight held-out rescue tasks. Scenes 1–5 use visible survivors; scenes 6–8 require
          proximity-based detection before the reach bonus is achievable.
        </p>

        <div className="report-table-wrap">
          <table className="report-table">
            <caption className="report-caption">Table I — Fixed disaster benchmark tasks</caption>
            <thead>
              <tr>
                <th>#</th>
                <th>Scene ID</th>
                <th>Difficulty</th>
                <th>Hazards</th>
                <th>Buried</th>
                <th>Signal</th>
                <th>Objective</th>
              </tr>
            </thead>
            <tbody>
              {BENCHMARK_SCENES.map((scene) => (
                <tr key={scene.id}>
                  <td className="mono">{scene.id}</td>
                  <td className="mono">{scene.name}</td>
                  <td>
                    <span className={`report-pill report-pill--${scene.difficulty}`}>
                      {scene.difficulty}
                    </span>
                  </td>
                  <td className="mono">{scene.hazards}</td>
                  <td>{scene.buried ? 'yes' : '—'}</td>
                  <td className="mono">{scene.signal ?? '—'}</td>
                  <td>{scene.objective}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Section 05 — MDP specification */}
      <section className="landing-section report-section">
        <div className="landing-section-hd">
          <h2>
            MDP Specification <span>— Observation & Reward</span>
          </h2>
          <span className="sec-idx mono">[ SECTION 05 / SPEC ]</span>
        </div>

        <div className="report-two-col">
          <div>
            <p className="report-lede">
              Table II decomposes the {MDP_CONSTANTS.obsDim}D observation vector returned by{' '}
              <code>DisasterEnv._get_obs()</code>.
            </p>
            <div className="report-table-wrap">
              <table className="report-table report-table--compact">
                <caption className="report-caption">Table II — Observation vector (21D)</caption>
                <thead>
                  <tr>
                    <th>Idx</th>
                    <th>D</th>
                    <th>Symbol</th>
                    <th>Description</th>
                  </tr>
                </thead>
                <tbody>
                  {OBSERVATION_SPEC.map((row) => (
                    <tr key={row.index}>
                      <td className="mono">{row.index}</td>
                      <td className="mono">{row.dims}</td>
                      <td className="mono">{row.symbol}</td>
                      <td>{row.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div>
            <p className="report-lede">
              Table III summarizes per-step reward terms. Terminal reach and first-detection bonuses apply on
              discrete events.
            </p>
            <div className="report-table-wrap">
              <table className="report-table report-table--compact">
                <caption className="report-caption">Table III — Reward decomposition</caption>
                <thead>
                  <tr>
                    <th>Term</th>
                    <th>Formula</th>
                    <th>Notes</th>
                  </tr>
                </thead>
                <tbody>
                  {REWARD_TERMS.map((row) => (
                    <tr key={row.term}>
                      <td>{row.term}</td>
                      <td className="mono report-formula">{row.formula}</td>
                      <td>{row.notes}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <div className="report-spec-box">
          <h3 className="report-spec-title">Training configuration</h3>
          <dl className="report-spec-dl">
            <div>
              <dt>Action space</dt>
              <dd>{MDP_CONSTANTS.actionSpace}</dd>
            </div>
            <div>
              <dt>Algorithm</dt>
              <dd>{MDP_CONSTANTS.algorithm}</dd>
            </div>
            <div>
              <dt>Policy network</dt>
              <dd>{MDP_CONSTANTS.policy}</dd>
            </div>
            <div>
              <dt>Total timesteps</dt>
              <dd>{MDP_CONSTANTS.trainingSteps}</dd>
            </div>
            <div>
              <dt>Parallel envs</dt>
              <dd>{MDP_CONSTANTS.parallelEnvs} (one per scene)</dd>
            </div>
            <div>
              <dt>World bounds</dt>
              <dd>{MDP_CONSTANTS.worldBounds}</dd>
            </div>
            <div>
              <dt>Train / eval horizon</dt>
              <dd>
                {MDP_CONSTANTS.maxStepsTrain} / {MDP_CONSTANTS.maxStepsEval} steps
              </dd>
            </div>
            <div>
              <dt>Success criterion</dt>
              <dd>distance &lt; {MDP_CONSTANTS.reachTolerance}</dd>
            </div>
          </dl>
        </div>
      </section>

      {/* Section 06 — Eval results & rollouts */}
      <section className="landing-section report-section">
        <div className="landing-section-hd">
          <h2>
            Evaluation Results <span>— Held-out Rollouts</span>
          </h2>
          <span className="sec-idx mono">[ SECTION 06 / EVAL ]</span>
        </div>

        <p className="report-lede">
          Table IV reports deterministic evaluation rollouts exported via{' '}
          <code>export_run.py</code>. GIFs are served from <code>runs/</code> when the backend is online.
          {runMeta && (
            <>
              {' '}
              View live runs in{' '}
              <Link href="/mission-control" className="report-inline-link">
                Mission Control
              </Link>
              .
            </>
          )}
        </p>

        <div className="report-table-wrap">
          <table className="report-table">
            <caption className="report-caption">
              Table IV — Scene-level evaluation
              {runMeta ? ` (${runMeta.name})` : ''}
            </caption>
            <thead>
              <tr>
                <th>#</th>
                <th>Scene</th>
                <th>Reached</th>
                <th>Detected</th>
                <th>Steps</th>
                <th>Total reward</th>
                <th>Rollout</th>
              </tr>
            </thead>
            <tbody>
              {BENCHMARK_SCENES.map((scene, idx) => {
                const evalRow = evalByName.get(scene.name);
                const reached = evalRow?.reached ?? false;
                const detected = evalRow?.detection_event ?? false;
                const url = runMeta ? gifUrl(runMeta.id, evalRow?.gif) : null;

                return (
                  <tr key={scene.name}>
                    <td className="mono">{String(idx).padStart(2, '0')}</td>
                    <td className="mono">{scene.name}</td>
                    <td>
                      <span className={`report-pill ${reached ? 'report-pill--ok' : 'report-pill--fail'}`}>
                        {evalRow ? (reached ? 'yes' : 'no') : '—'}
                      </span>
                    </td>
                    <td>
                      {scene.buried ? (
                        <span
                          className={`report-pill ${detected ? 'report-pill--ok' : 'report-pill--fail'}`}
                        >
                          {evalRow ? (detected ? 'yes' : 'no') : '—'}
                        </span>
                      ) : (
                        'n/a'
                      )}
                    </td>
                    <td className="mono">{evalRow?.steps ?? '—'}</td>
                    <td className="mono">
                      {evalRow?.total_reward != null ? evalRow.total_reward.toFixed(2) : '—'}
                    </td>
                    <td>
                      {url ? (
                        <a href={url} target="_blank" rel="noopener noreferrer" className="report-inline-link">
                          GIF
                        </a>
                      ) : (
                        '—'
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {runMeta && evalData?.scenes?.some((s) => s.gif) && (
          <div className="report-figures">
            <h3 className="report-figures-title">Figure 1 — Evaluation rollout frames</h3>
            <div className="report-figure-grid">
              {BENCHMARK_SCENES.map((scene) => {
                const evalRow = evalByName.get(scene.name);
                const url = gifUrl(runMeta.id, evalRow?.gif);
                if (!url) return null;
                return (
                  <figure key={scene.name} className="report-figure-card">
                    <img
                      src={url}
                      alt={`Rollout for ${formatSceneName(scene.name)}`}
                      loading="lazy"
                      className="report-figure-img"
                    />
                    <figcaption>
                      <span className="mono">{scene.name}</span>
                      <span
                        className={`report-pill report-pill--sm ${
                          evalRow?.reached ? 'report-pill--ok' : 'report-pill--fail'
                        }`}
                      >
                        {evalRow?.reached ? 'reached' : 'timeout'}
                      </span>
                    </figcaption>
                  </figure>
                );
              })}
            </div>
          </div>
        )}
      </section>

      {/* Section 07 — Research backlog */}
      <section className="landing-section report-section">
        <div className="landing-section-hd">
          <h2>
            Research Backlog <span>— Open Work Items</span>
          </h2>
          <span className="sec-idx mono">[ SECTION 07 / BACKLOG ]</span>
        </div>

        <p className="report-lede">
          Table V tracks engineering and research tasks tied to the current evaluation pipeline. Status reflects
          the <code>landing-improvements</code> branch as of the latest export.
        </p>

        <div className="report-table-wrap">
          <table className="report-table">
            <caption className="report-caption">Table V — Project task tracker</caption>
            <thead>
              <tr>
                <th>ID</th>
                <th>Area</th>
                <th>Status</th>
                <th>Description</th>
              </tr>
            </thead>
            <tbody>
              {RESEARCH_TASKS.map((task) => (
                <tr key={task.id}>
                  <td className="mono">{task.id}</td>
                  <td>{task.area}</td>
                  <td>
                    <span className={`report-pill report-pill--${task.status}`}>
                      {STATUS_LABEL[task.status]}
                    </span>
                  </td>
                  <td>{task.description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}
