'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import LandingReport from './LandingReport';
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
            <div className="brand-sub mono">EPISODE RUNNER · v0.4.2</div>
          </div>
        </Link>
        <WorkflowNav active="overview" />
        <div className="clock">
          <span className="lbl">UTC</span>
          <span className="val mono">{clockUtc}</span>
          <span className="lbl" style={{ marginLeft: 10 }}>STATUS</span>
          <span className="val mono" style={{ color: 'var(--ok)' }}>ONLINE</span>
        </div>
      </header>

      {/* Hero Section */}
      <header className="landing-hero" style={{ borderBottom: '1px solid var(--rule)' }}>
        <h1>
          PROJECT <span>BATTLE ANGEL</span>
        </h1>
        <p style={{ fontStyle: 'italic', fontFamily: "'Newsreader', Georgia, serif", fontSize: '20px' }}>
          Gemini-Guided Ground Rescue Robotics in 3D Space
        </p>
        <p>
          A state-of-the-art robotic agent that navigates a 3D simulated disaster environment.
          By combining natural language grounding through Google Gemini with a deep reinforcement learning policy
          trained in MuJoCo, the agent makes real-time decisions to prioritize and rescue survivors.
        </p>
        <div className="hero-cta-group">
          <Link className="btn-primary" href="/mission-control">
            Enter Training Gym
          </Link>
          <Link className="btn-secondary" href="/generate">
            Generate a Scene
          </Link>
        </div>
      </header>

      <section className="workflow-rail" aria-label="Policy deployment workflow">
        <div className="workflow-rail-heading mono">POLICY PATH / SIMULATION TO DEPLOYMENT</div>
        <Link href="/mission-control" className="workflow-stage">
          <span className="workflow-step mono">01</span>
          <strong>Training Gym</strong>
          <span>Train and validate PPO across eight environments.</span>
        </Link>
        <Link href="/generate" className="workflow-stage">
          <span className="workflow-step mono">02</span>
          <strong>Scene Generator</strong>
          <span>Create candidate real-world rescue scenes.</span>
        </Link>
        <Link href="/console" className="workflow-stage">
          <span className="workflow-step mono">03</span>
          <strong>Interactive Console</strong>
          <span>Visualize the policy response and inspect outcomes.</span>
        </Link>
      </section>

      {/* System Architecture Bento Section */}
      <section className="landing-section">
        <div className="landing-section-hd">
          <h2>
            System Pillars <span>— Architectural Breakdown</span>
          </h2>
          <span className="sec-idx mono">[ SECTION 02 / ARCHITECTURE ]</span>
        </div>

        <div className="bento-grid">
          <div className="bento-card">
            <div>
              <h3>
                Gemini Goal Selector
                <span className="badge">NLP GROUNDING</span>
              </h3>
              <p>
                Accepts unstructured, natural language operator orders (e.g. <i>&quot;save the child first, ignore the adult&quot;</i>).
                An integrated <b>Gemini 1.5 Flash</b> model parses the text against a structured JSON schema,
                extracting the correct target coordinate profile while providing fallback safeguards to guarantee stable performance.
              </p>
            </div>
            <div style={{ marginTop: '20px', fontSize: '11px', color: 'var(--red)', fontFamily: 'monospace' }}>
              schema: &#123; target_id: &quot;child&quot; | &quot;adult&quot;, confidence: float, reason: string &#125;
            </div>
          </div>

          <div className="bento-card">
            <div>
              <h3>
                RL Navigation Controller
                <span className="badge">PPO POLICY</span>
              </h3>
              <p>
                A high-frequency <b>PPO (Proximal Policy Optimization)</b> model trained for 500,000 steps using Stable-Baselines3.
                Running on a continuous 2D-force action space, it accepts a relative 10-dimensional coordinate vector to navigate around debris,
                obstacle boxes, and chemical hazards to guide the robot to the chosen target survivor.
              </p>
            </div>
            <div style={{ marginTop: '20px', fontSize: '11px', color: 'var(--ink-dim)', fontFamily: 'monospace' }}>
              policy network: MLP (256x256) · batch: 256 · gamma: 0.99
            </div>
          </div>

          <div className="bento-card">
            <div>
              <h3>
                3D Disaster Environment
                <span className="badge">MuJoCo PHYSICS</span>
              </h3>
              <p>
                A structured <code>DisasterEnv</code> representing warehouse sector D-14. Renders a detailed Unitree G1 humanoid
                robot model. Includes dynamic floor grids, static concrete obstacles, and circular chemical hazard zones
                enforced by collision-penalties and custom step rewards to encourage smart routing.
              </p>
            </div>
            <div style={{ marginTop: '20px', fontSize: '11px', color: 'var(--ink-dim)', fontFamily: 'monospace' }}>
              arena bounds: 16m x 16m · reach tolerance: 0.75m · step budget: 150
            </div>
          </div>

          <div className="bento-card">
            <div>
              <h3>
                Dual-Process Architecture
                <span className="badge">macOS CONFLATION</span>
              </h3>
              <p>
                Bypasses macOS thread deadlocks by separating the <b>FastAPI event loop</b> and the <b>MuJoCo simulation viewer</b>
                into distinct multiprocessing boundaries. A communication queue coordinates goal injections, and a shared manager
                dictionary feeds real-time physical simulation telemetry back to the Web UI.
              </p>
            </div>
            <div style={{ marginTop: '20px', fontSize: '11px', color: 'var(--red)', fontFamily: 'monospace' }}>
              robot_process.py (Viewer thread) &lt;--- queue/manager ---&gt; server.py (FastAPI)
            </div>
          </div>
        </div>
      </section>

      <LandingReport />

      {/* Training & Model Suite Section */}
      <section className="landing-section report-section">
        <div className="landing-section-hd">
          <h2>
            Training Gym Archive <span>— PPO Progress Metrics</span>
          </h2>
          <span className="sec-idx mono">[ SECTION 08 / METRICS ]</span>
        </div>

        <TrainingRuns />
      </section>

      {/* Footer */}
      <footer className="landing-footer">
        <p>© 2026 Battle Angel Robotics · Hackathon Project Ground Rescue</p>
      </footer>
    </div>
  );
}
