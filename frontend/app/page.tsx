'use client';

import Console from './Console';
import TrainingRuns from './TrainingRuns';

export default function Page() {
  const scrollToConsole = () => {
    const el = document.getElementById('mission-control');
    if (el) {
      el.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <div style={{ background: 'var(--bg)', minHeight: '100vh', color: 'var(--ink)' }}>
      {/* Hero Section */}
      <header className="landing-hero">
        <div style={{ display: 'inline-block', padding: '2px 8px', border: '1px solid var(--red)', color: 'var(--red)', fontSize: '11px', fontWeight: 600, letterSpacing: '0.15em', textTransform: 'uppercase', marginBottom: '24px', fontStyle: 'normal' }}>
          HACKATHON DEMO · ACTIVE DEVELOPMENT
        </div>
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
          <button className="btn-primary" onClick={scrollToConsole}>
            Launch Mission Control
          </button>
          <a 
            className="btn-secondary" 
            href="https://github.com/aravindkrishna2008/disaster-rescue" 
            target="_blank" 
            rel="noopener noreferrer"
          >
            Repository
          </a>
        </div>
      </header>

      {/* Main Console Section */}
      <section id="mission-control" className="landing-section" style={{ paddingBottom: '30px' }}>
        <div className="landing-section-hd">
          <h2>
            Mission Control Console <span>— Real-time WebGL Telemetry</span>
          </h2>
          <span className="sec-idx mono">[ SECTION 01 / TELEMETRY ]</span>
        </div>
        
        <div className="dashboard-container">
          <Console />
        </div>
      </section>

      {/* System Architecture Bento Section */}
      <section className="landing-section" style={{ paddingTop: '30px', paddingBottom: '30px' }}>
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
                Accepts unstructured, natural language operator orders (e.g. <i>"save the child first, ignore the adult"</i>). 
                An integrated **Gemini 1.5 Flash** model parses the text against a structured JSON schema, 
                extracting the correct target coordinate profile while providing fallback safeguards to guarantee stable performance.
              </p>
            </div>
            <div style={{ marginTop: '20px', fontSize: '11px', color: 'var(--red)', fontFamily: 'monospace' }}>
              schema: &#123; target_id: "child" | "adult", confidence: float, reason: string &#125;
            </div>
          </div>

          <div className="bento-card">
            <div>
              <h3>
                RL Navigation Controller
                <span className="badge">PPO POLICY</span>
              </h3>
              <p>
                A high-frequency **PPO (Proximal Policy Optimization)** model trained for 500,000 steps using Stable-Baselines3. 
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
                A structured `DisasterEnv` representing warehouse sector D-14. Renders a detailed Unitree G1 humanoid 
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
                Bypasses macOS thread deadlocks by separating the **FastAPI event loop** and the **MuJoCo simulation viewer** 
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

      {/* Training & Model Suite Section */}
      <section className="landing-section" style={{ paddingTop: '30px' }}>
        <div className="landing-section-hd">
          <h2>
            Training Pipeline <span>— Progress Metrics</span>
          </h2>
          <span className="sec-idx mono">[ SECTION 03 / METRICS ]</span>
        </div>

        <TrainingRuns />
      </section>

      {/* Footer */}
      <footer className="landing-footer">
        <p>© 2026 Battle Angel Robotics · Hackathon Project Ground Rescue</p>
        <p>
          Powered by{' '}
          <a href="https://deepmind.google/technologies/gemini/" target="_blank" rel="noopener noreferrer">
            Gemini 1.5 Flash
          </a>{' '}
          ·{' '}
          <a href="https://stable-baselines3.readthedocs.io/" target="_blank" rel="noopener noreferrer">
            Stable-Baselines3
          </a>{' '}
          ·{' '}
          <a href="https://threejs.org/" target="_blank" rel="noopener noreferrer">
            Three.js
          </a>
        </p>
      </footer>
    </div>
  );
}
