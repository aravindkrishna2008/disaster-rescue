'use client';

import Console from './Console';

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

        <div className="metrics-section">
          {/* Left: Directory structure */}
          <div>
            <h3 style={{ fontSize: '16px', fontWeight: 600, margin: '0 0 16px', fontStyle: 'italic' }}>
              Training Run File Pipeline
            </h3>
            <p style={{ fontSize: '14px', lineHeight: 1.6, color: 'var(--ink-mid)', marginBottom: '20px' }}>
              To ensure training reproducibility and model comparison post-hackathon, 
              each run exports evaluation scores, model checkpoints, and configuration parameters in a structured format:
            </p>
            
            <div className="runs-box">
              <div>runs/</div>
              <div>&nbsp;&nbsp;└── <b>ppo_fixed_six_final/</b></div>
              <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── <span className="dim">summary.json</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span className="dim"># Run parameters &amp; meta</span></div>
              <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── <span className="dim">eval.json</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span className="dim"># 6-scene test results</span></div>
              <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── <b>checkpoints/</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span className="dim"># Models at step checkpoints</span></div>
              <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├── <span className="dim">ppo_fixed_six_120000_steps.zip</span></div>
              <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└── <span className="dim">ppo_fixed_six_480000_steps.zip</span></div>
              <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── <span className="dim">tb_logs/</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span className="dim"># TensorBoard diagnostics</span></div>
            </div>
          </div>

          {/* Right: SVG reward chart */}
          <div className="chart-box">
            <h4>PPO Policy Learning Curve (Cumulative Reward)</h4>
            <div className="chart-container">
              <svg viewBox="0 0 400 200" width="100%" height="100%">
                {/* Grid lines */}
                <line x1="30" y1="20" x2="380" y2="20" stroke="#d8d2c4" strokeDasharray="3,3" />
                <line x1="30" y1="70" x2="380" y2="70" stroke="#d8d2c4" strokeDasharray="3,3" />
                <line x1="30" y1="120" x2="380" y2="120" stroke="#d8d2c4" strokeDasharray="3,3" />
                <line x1="30" y1="170" x2="380" y2="170" stroke="#c2bbac" />
                
                {/* X Axis division */}
                <line x1="30" y1="170" x2="30" y2="15" stroke="#c2bbac" />
                <line x1="120" y1="170" x2="120" y2="175" stroke="#c2bbac" />
                <line x1="210" y1="170" x2="210" y2="175" stroke="#c2bbac" />
                <line x1="300" y1="170" x2="300" y2="175" stroke="#c2bbac" />
                <line x1="380" y1="170" x2="380" y2="175" stroke="#c2bbac" />

                {/* X labels */}
                <text x="30" y="190" fontSize="9" fill="#837c6f" textAnchor="middle" fontFamily="monospace">0k</text>
                <text x="120" y="190" fontSize="9" fill="#837c6f" textAnchor="middle" fontFamily="monospace">120k</text>
                <text x="210" y="190" fontSize="9" fill="#837c6f" textAnchor="middle" fontFamily="monospace">240k</text>
                <text x="300" y="190" fontSize="9" fill="#837c6f" textAnchor="middle" fontFamily="monospace">360k</text>
                <text x="380" y="190" fontSize="9" fill="#837c6f" textAnchor="middle" fontFamily="monospace">500k</text>

                {/* Y labels */}
                <text x="22" y="173" fontSize="9" fill="#837c6f" textAnchor="end" fontFamily="monospace">-250</text>
                <text x="22" y="123" fontSize="9" fill="#837c6f" textAnchor="end" fontFamily="monospace">-150</text>
                <text x="22" y="73" fontSize="9" fill="#837c6f" textAnchor="end" fontFamily="monospace">-50</text>
                <text x="22" y="23" fontSize="9" fill="#837c6f" textAnchor="end" fontFamily="monospace">+50</text>

                {/* Training Curve Line */}
                <path 
                  d="M 30 170 Q 60 160, 90 145 T 150 120 T 210 80 T 280 62 T 340 50 T 380 44" 
                  fill="none" 
                  stroke="#b02e26" 
                  strokeWidth="2.5" 
                />
                
                {/* Dotted threshold line */}
                <line x1="30" y1="44" x2="380" y2="44" stroke="#2a5c3a" strokeWidth="1" strokeDasharray="4,2" />
                <text x="375" y="38" fontSize="8" fill="#2a5c3a" textAnchor="end" fontWeight="600" fontFamily="monospace">SOLVED THRESHOLD</text>
              </svg>
            </div>
            <div style={{ fontSize: '11px', color: 'var(--ink-dim)', fontStyle: 'italic', marginTop: '10px', textAlign: 'center' }}>
              * Mean episode reward trends positive and converges near step 360k.
            </div>
          </div>
        </div>
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
