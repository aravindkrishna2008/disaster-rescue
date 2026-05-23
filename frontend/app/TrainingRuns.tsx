'use client';

import { useEffect, useMemo, useState } from 'react';
import {
  TRAINING_RUNS,
  curveToPath,
  formatSteps,
  rewardToY,
  stepToX,
  xAxisTicks,
  yAxisTicks,
  type TrainingRun,
} from './trainingRunData';

type RunListItem = { id: string; name: string; subtitle?: string };

function mapApiRun(data: Record<string, unknown>): TrainingRun {
  return {
    id: String(data.id),
    name: String(data.name),
    subtitle: String(data.subtitle ?? ''),
    totalSteps: Number(data.totalSteps ?? 0),
    solvedThreshold: Number(data.solvedThreshold ?? 0),
    yMin: Number(data.yMin ?? -250),
    yMax: Number(data.yMax ?? 50),
    caption: String(data.caption ?? ''),
    curve: (data.curve as TrainingRun['curve']) ?? [],
    checkpoints: (data.checkpoints as string[]) ?? [],
    fileTree: (data.fileTree as string[]) ?? [],
    advancedStats: (data.advancedStats as TrainingRun['advancedStats']) ?? [],
    source: data.source === 'runs' ? 'runs' : 'static',
    eval: data.eval as TrainingRun['eval'],
    exportedAt: data.exportedAt as string | undefined,
  };
}

export default function TrainingRuns() {
  const [runs, setRuns] = useState<TrainingRun[]>(TRAINING_RUNS);
  const [selectedId, setSelectedId] = useState(TRAINING_RUNS[0].id);
  const [loading, setLoading] = useState(true);
  const [dataSource, setDataSource] = useState<'runs' | 'static'>('static');

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const listRes = await fetch('/runs');
        if (!listRes.ok) throw new Error(`HTTP ${listRes.status}`);
        const { runs: items } = (await listRes.json()) as { runs: RunListItem[] };
        if (!items?.length) throw new Error('no exported runs');

        const manifests = await Promise.all(
          items.map(async (item) => {
            const r = await fetch(`/runs/${item.id}`);
            if (!r.ok) throw new Error(`run ${item.id}: HTTP ${r.status}`);
            return mapApiRun(await r.json());
          }),
        );

        if (!cancelled && manifests.length > 0) {
          setRuns(manifests);
          setSelectedId(manifests[0].id);
          setDataSource('runs');
        }
      } catch {
        if (!cancelled) {
          setRuns(TRAINING_RUNS);
          setSelectedId(TRAINING_RUNS[0].id);
          setDataSource('static');
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

  const run = useMemo(
    () => runs.find((r) => r.id === selectedId) ?? runs[0],
    [runs, selectedId],
  );

  const pathD = curveToPath(run.curve, run.totalSteps, run.yMin, run.yMax);
  const thresholdY = rewardToY(run.solvedThreshold, run.yMin, run.yMax);
  const xTicks = xAxisTicks(run.totalSteps);
  const yTicks = yAxisTicks(run.yMin, run.yMax);

  const statsByGroup = useMemo(() => {
    const groups = new Map<string, typeof run.advancedStats>();
    for (const stat of run.advancedStats) {
      const g = stat.group ?? 'General';
      if (!groups.has(g)) groups.set(g, []);
      groups.get(g)!.push(stat);
    }
    return groups;
  }, [run]);

  const evalBadge =
    run.eval && run.eval.scene_count != null
      ? `${run.eval.success_count ?? 0} / ${run.eval.scene_count} scenes reached`
      : null;

  return (
    <div className="metrics-section">
      <div>
        <div className="training-run-header">
          <h3>Training Run File Pipeline</h3>
          <label className="training-run-select-wrap">
            <span className="training-run-select-label">Run</span>
            <select
              className="training-run-select"
              value={selectedId}
              onChange={(e) => setSelectedId(e.target.value)}
              aria-label="Select training run"
              disabled={loading}
            >
              {runs.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.name}
                </option>
              ))}
            </select>
          </label>
        </div>

        <p className="training-run-subtitle">
          {run.subtitle}
          {dataSource === 'runs' && (
            <span className="training-run-live-tag"> · live from runs/</span>
          )}
          {dataSource === 'static' && !loading && (
            <span className="training-run-fallback-tag"> · fallback (run export_run.py)</span>
          )}
        </p>
        {evalBadge && <p className="training-run-eval-badge">{evalBadge}</p>}
        <p className="training-run-desc">
          Each run exports evaluation scores, model checkpoints, and configuration parameters in a
          structured format for reproducibility and comparison.
        </p>

        <div className="runs-box">
          {run.fileTree.map((line, i) => (
            <div key={`${run.id}-${i}`}>{line}</div>
          ))}
        </div>

        <details className="advanced-stats">
          <summary>Advanced statistics</summary>
          <div className="advanced-stats-body">
            {[...statsByGroup.entries()].map(([group, stats]) => (
              <div key={group} className="stats-group">
                <h4>{group}</h4>
                <dl className="stats-grid">
                  {stats.map((s) => (
                    <div key={s.label} className="stat-row">
                      <dt>{s.label}</dt>
                      <dd>{s.value}</dd>
                    </div>
                  ))}
                </dl>
              </div>
            ))}
          </div>
        </details>
      </div>

      <div className="chart-box">
        <div className="chart-box-hd">
          <h4>PPO Policy Learning Curve (Cumulative Reward)</h4>
          <span className="chart-run-tag">{run.name}</span>
        </div>
        <div className="chart-container">
          {run.curve.length === 0 ? (
            <p className="chart-empty">No TensorBoard curve exported for this run.</p>
          ) : (
            <svg viewBox="0 0 400 200" width="100%" height="100%" role="img" aria-label={`Learning curve for ${run.name}`}>
              {yTicks.map((tick) => {
                const y = rewardToY(tick, run.yMin, run.yMax);
                return (
                  <line
                    key={`y-${tick}`}
                    x1="30"
                    y1={y}
                    x2="380"
                    y2={y}
                    stroke={tick === run.yMin ? '#c2bbac' : '#d8d2c4'}
                    strokeDasharray={tick === run.yMin ? undefined : '3,3'}
                  />
                );
              })}

              <line x1="30" y1="170" x2="30" y2="15" stroke="#c2bbac" />
              <line x1="30" y1="170" x2="380" y2="170" stroke="#c2bbac" />

              {xTicks.map((tick) => (
                <g key={`x-${tick}`}>
                  <line x1={stepToX(tick, run.totalSteps)} y1="170" x2={stepToX(tick, run.totalSteps)} y2="175" stroke="#c2bbac" />
                  <text
                    x={stepToX(tick, run.totalSteps)}
                    y="190"
                    fontSize="9"
                    fill="#837c6f"
                    textAnchor="middle"
                    fontFamily="inherit"
                  >
                    {formatSteps(tick)}
                  </text>
                </g>
              ))}

              {yTicks.map((tick) => (
                <text
                  key={`yl-${tick}`}
                  x="22"
                  y={rewardToY(tick, run.yMin, run.yMax) + 3}
                  fontSize="9"
                  fill="#837c6f"
                  textAnchor="end"
                  fontFamily="inherit"
                >
                  {tick > 0 ? `+${tick}` : tick}
                </text>
              ))}

              <path d={pathD} fill="none" stroke="#b02e26" strokeWidth="2.5" strokeLinejoin="round" />

              <line
                x1="30"
                y1={thresholdY}
                x2="380"
                y2={thresholdY}
                stroke="#2a5c3a"
                strokeWidth="1"
                strokeDasharray="4,2"
              />
              <text x="375" y={thresholdY - 6} fontSize="8" fill="#2a5c3a" textAnchor="end" fontWeight="600" fontFamily="inherit">
                SOLVED THRESHOLD
              </text>
            </svg>
          )}
        </div>
        <p className="chart-caption">* {run.caption}</p>
      </div>
    </div>
  );
}
