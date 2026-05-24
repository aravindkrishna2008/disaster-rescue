'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { BUILD_PHASES, type BuildPhase } from './landingStoryData';

type RunSnapshot = {
  id: string;
  name: string;
  totalSteps?: number;
  obsDim?: number;
  gitCommit?: string;
  exportedAt?: string;
  successCount?: number;
  sceneCount?: number;
  meanReward?: number;
};

async function fetchRunSnapshot(runId: string): Promise<RunSnapshot | null> {
  try {
    const res = await fetch(`/runs/${runId}`);
    if (!res.ok) return null;
    const data = (await res.json()) as Record<string, unknown>;
    const evalData = data.eval as Record<string, unknown> | undefined;
    return {
      id: runId,
      name: String(data.name ?? runId),
      totalSteps: data.total_steps as number | undefined,
      obsDim: data.obs_dim as number | undefined,
      gitCommit: data.git_commit as string | undefined,
      exportedAt: data.exported_at as string | undefined,
      successCount: evalData?.success_count as number | undefined,
      sceneCount: evalData?.scene_count as number | undefined,
      meanReward: evalData?.mean_reward as number | undefined,
    };
  } catch {
    return null;
  }
}

function formatSteps(n?: number) {
  if (n == null) return '—';
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${Math.round(n / 1_000)}k`;
  return String(n);
}

function PhaseBlock({
  phase,
  snapshot,
}: {
  phase: BuildPhase;
  snapshot: RunSnapshot | null;
}) {
  return (
    <article className="story-phase" id={phase.id}>
      <div className="story-phase-rail">
        <span className="story-phase-index mono">{phase.index}</span>
      </div>

      <div className="story-phase-body">
        <header className="story-phase-header">
          <p className="story-phase-eyebrow mono">PHASE {phase.index}</p>
          <h3>{phase.title}</h3>
          <p className="story-phase-headline">{phase.headline}</p>
        </header>

        {phase.screenshot && (
          <figure className="story-phase-screenshot">
            <img
              src={phase.screenshot.src}
              alt={phase.screenshot.alt}
              loading="lazy"
              className="story-phase-screenshot-img"
            />
            <figcaption className="mono">{phase.screenshot.caption}</figcaption>
          </figure>
        )}

        <div className="story-phase-prose">
            {phase.story.map((para) => (
              <p key={para.slice(0, 40)}>{para}</p>
            ))}

            <h4 className="story-deliverables-title mono">Why it matters for robotics</h4>
            <ul className="story-deliverables">
              {phase.deliverables.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>

            {phase.runId && snapshot && (
              <dl className="story-run-stats">
                <div>
                  <dt className="mono">Run</dt>
                  <dd className="mono">{snapshot.name}</dd>
                </div>
                <div>
                  <dt className="mono">Steps</dt>
                  <dd className="mono">{formatSteps(snapshot.totalSteps)}</dd>
                </div>
                <div>
                  <dt className="mono">Obs</dt>
                  <dd className="mono">{snapshot.obsDim != null ? `${snapshot.obsDim}D` : '—'}</dd>
                </div>
                <div>
                  <dt className="mono">Reach</dt>
                  <dd className="mono">
                    {snapshot.successCount != null && snapshot.sceneCount != null
                      ? `${snapshot.successCount}/${snapshot.sceneCount}`
                      : '—'}
                  </dd>
                </div>
                <div>
                  <dt className="mono">Mean reward</dt>
                  <dd className="mono">
                    {snapshot.meanReward != null ? snapshot.meanReward.toFixed(0) : '—'}
                  </dd>
                </div>
                <div>
                  <dt className="mono">Commit</dt>
                  <dd className="mono">{snapshot.gitCommit?.slice(0, 7) ?? '—'}</dd>
                </div>
              </dl>
            )}

            {phase.showCta !== false && (
              <Link href={phase.href} className="btn-primary story-phase-cta">
                {phase.linkLabel}
              </Link>
            )}
          </div>
      </div>
    </article>
  );
}

export default function LandingStory() {
  const [snapshots, setSnapshots] = useState<Record<string, RunSnapshot | null>>({});

  useEffect(() => {
    let cancelled = false;

    async function load() {
      const runIds = [...new Set(BUILD_PHASES.map((p) => p.runId).filter(Boolean) as string[])];
      const results = await Promise.all(runIds.map((id) => fetchRunSnapshot(id)));
      if (cancelled) return;
      const map: Record<string, RunSnapshot | null> = {};
      runIds.forEach((id, i) => {
        map[id] = results[i];
      });
      setSnapshots(map);
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <section className="landing-section story-section" aria-labelledby="build-story-title">
      <div className="landing-section-hd">
        <h2 id="build-story-title">
          The Stack <span>— Three Tools for Rescue Robotics</span>
        </h2>
        <span className="sec-idx mono">[ WORKFLOW ]</span>
      </div>

      <p className="story-intro">
        Battle Angel is three interfaces on one MuJoCo + PPO backbone: train policies in the Gym,
        generate new disaster layouts on demand, and drive the robot from the Console with natural
        language.
      </p>

      <div className="story-timeline">
        {BUILD_PHASES.map((phase) => (
          <PhaseBlock
            key={phase.id}
            phase={phase}
            snapshot={phase.runId ? snapshots[phase.runId] ?? null : null}
          />
        ))}
      </div>
    </section>
  );
}
