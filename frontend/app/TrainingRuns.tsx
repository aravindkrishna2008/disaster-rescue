'use client';

import { useMemo, useState } from 'react';
import {
  TRAINING_RUNS,
  curveToPath,
  formatSteps,
  rewardToY,
  stepToX,
  xAxisTicks,
  yAxisTicks,
} from './trainingRunData';

export default function TrainingRuns() {
  const [selectedId, setSelectedId] = useState(TRAINING_RUNS[0].id);

  const run = useMemo(
    () => TRAINING_RUNS.find((r) => r.id === selectedId) ?? TRAINING_RUNS[0],
    [selectedId],
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
            >
              {TRAINING_RUNS.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.name}
                </option>
              ))}
            </select>
          </label>
        </div>

        <p className="training-run-subtitle">{run.subtitle}</p>
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
        </div>
        <p className="chart-caption">* {run.caption}</p>
      </div>
    </div>
  );
}
