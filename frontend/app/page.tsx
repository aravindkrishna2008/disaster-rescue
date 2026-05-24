'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import LandingReport from './LandingReport';
import LandingStory from './LandingStory';
import TrainingRuns from './TrainingRuns';
import WorkflowNav from './WorkflowNav';

const tsNow = () => {
  const d = new Date();
  return [d.getUTCHours(), d.getUTCMinutes(), d.getUTCSeconds()]
    .map((v) => String(v).padStart(2, '0'))
    .join(':');
};

export default function Page() {
  const [clockUtc, setClockUtc] = useState('--:--:--');

  useEffect(() => {
    setClockUtc(tsNow());
    const id = setInterval(() => setClockUtc(tsNow()), 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <div style={{ background: 'var(--bg)', minHeight: '100vh', color: 'var(--ink)' }}>
      <header className="topbar">
        <Link href="/" className="brand">
          <div className="brand-mark" aria-hidden="true"></div>
          <div>
            <div className="brand-name"><em>Battle Angel</em></div>
          </div>
        </Link>
        <WorkflowNav active="overview" />
        <div className="clock">
          <span className="lbl">UTC</span>
          <span className="val mono">{clockUtc}</span>
        </div>
      </header>

      <header className="landing-hero landing-hero--story">
        <h1>
          <div className="brand-mark brand-mark--hero" aria-hidden="true"></div>
          <span>BATTLE ANGEL</span>
        </h1>
        <p className="landing-hero-body">
          Three tools for rescue robotics on a shared MuJoCo stack: the Training Gym benchmarks and
          trains locomotion policies, the Scene Generator produces new disaster layouts from language,
          and the Interactive Console lets operators set survivor priorities that Gemini 3.5 Flash
          turns into robot goals.
        </p>
        <div className="hero-cta-group">
          <Link className="btn-primary" href="/console">
            Try the Console
          </Link>
        </div>
      </header>

      <LandingStory />

      <div className="research-divider" id="research">
        <div className="research-divider-inner">
          <span className="research-divider-label mono">RESEARCH</span>
          <h2>Benchmark, MDP & Open Problems</h2>
          <p>
            Everything below is the paper-style side of the project — task definitions, observation
            and reward specs, held-out evaluation tables, learning curves, and the backlog we are
            still working through.
          </p>
        </div>
      </div>

      <LandingReport />

      <section className="landing-section report-section">
        <div className="landing-section-hd">
          <h2>
            Training Archive <span>— Learning Curves & Checkpoints</span>
          </h2>
          <span className="sec-idx mono">[ METRICS ]</span>
        </div>
        <TrainingRuns />
      </section>

      <footer className="landing-footer">
        <p>© 2026 Battle Angel Robotics · Gemini 3.5 Flash + MuJoCo PPO</p>
        <p className="landing-footer-links">
          <Link href="/mission-control">Training Gym</Link>
          <span aria-hidden="true"> · </span>
          <Link href="/generate">Scene Generator</Link>
          <span aria-hidden="true"> · </span>
          <Link href="/console">Console</Link>
        </p>
      </footer>
    </div>
  );
}
