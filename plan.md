# Setup Guide — Disaster Rescue Robot

**Time:** ~30 minutes  
**Goal:** Get the environment running and verify everything works

## ✅ Checklist

- [ ] Python 3.11+ installed
- [ ] Miniforge installed
- [ ] Conda environment created
- [ ] All packages installed
- [ ] MuJoCo smoke test passes
- [ ] Environment test passes
- [ ] Ready to run training

## Step 1: Install Miniforge (5 min)

Miniforge is a lightweight conda package manager.

### macOS (Apple Silicon M1/M2/M3/M4)
```bash
brew install --cask miniforge
# After install, restart your terminal
```

### macOS (Intel)
```bash
brew install --cask miniforge
```

### Linux (Ubuntu/Debian)
```bash
# Download and run installer
curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
```

### Windows
Download installer from: https://github.com/conda-forge/miniforge/releases

---

**Verify:** Open a new terminal and run:
```bash
conda --version
# Should output: conda X.XX.X
```

## Step 2: Clone Repository (2 min)

```bash
git clone https://github.com/your-org/disaster-rescue.git
cd disaster-rescue
```

If using existing repo, just pull latest:
```bash
cd disaster-rescue
git pull origin main
```

## Step 3: Create Conda Environment (5 min)

```bash
# Create Python 3.11 environment named "disaster"
conda create -n disaster python=3.11 -y

# Activate it
conda activate disaster

# You should see (disaster) in your terminal prompt
```

## Step 4: Install Dependencies (10 min)

With the `disaster` environment active, run:

```bash
pip install 'gymnasium[mujoco]' 'stable-baselines3==2.3.2' google-generativeai fastapi uvicorn 'imageio[ffmpeg]' torch tensorboard tqdm rich
```

**Note:** Replace `gymnasium[mujoco]` with `"gymnasium[mujoco]"` if your shell complains about brackets.

### What gets installed?
- `gymnasium[mujoco]` — Physics simulator + RL environment
- `stable-baselines3` — PPO algorithm
- `google-generativeai` — Gemini API (for hackathon day)
- `fastapi` + `uvicorn` — Web server (for hackathon day)
- `torch` — ML framework
- `tensorboard` — Training graphs
- `tqdm` + `rich` — Progress bars

**Install takes ~5 min.** Brew some coffee ☕

---

**Verify:** Run this to check all imports work:
```bash
python -c "import gymnasium; import mujoco; import stable_baselines3; print('✓ All imports OK')"
```

## Step 5: Smoke Test — MuJoCo (3 min)

MuJoCo is a physics simulator. Test it:

```bash
python -c "import mujoco; import mujoco.viewer; print('✓ MuJoCo imports OK')"
```

**Expected:** Should print `✓ MuJoCo imports OK` and return immediately.

### If it fails on macOS:
```bash
# Try reinstalling MuJoCo for Apple Silicon
pip uninstall mujoco -y
pip install mujoco
```

## Step 6: Smoke Test — DisasterEnv (2 min)

Test the custom environment:

```bash
python -c "
from disaster_env import DisasterEnv
env = DisasterEnv()
obs, _ = env.reset()
print(f'✓ Environment loaded')
print(f'  Observation shape: {obs.shape}')
print(f'  Observation: {obs}')
env.close()
"
```

**Expected output:**
```
✓ Environment loaded
  Observation shape: (7,)
  Observation: [-0.5  -0.5  5.0  5.0  7.07  0.0  0.0]
```

## Step 7: Quick Training Test (3 min)

Verify training works:

```bash
python -c "
import os
from disaster_env import DisasterEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

os.makedirs('./models', exist_ok=True)

def make_env():
    return DisasterEnv()

env = DummyVecEnv([make_env for _ in range(6)])
model = PPO('MlpPolicy', env, verbose=0, device='cpu')
print('Training for 2k steps...')
model.learn(total_timesteps=2000, progress_bar=True)
print('✓ Training works!')
"
```

**Expected:** Takes ~10 seconds, shows progress bar, prints success message.

## Step 8: Full Pre-Training (20 min)

This is the important one — run it before the hackathon!

```bash
python train.py
```

**What happens:**
- Trains PPO for 50,000 timesteps
- Shows live progress bar
- Saves checkpoints to `./models/ppo_model_*.zip`
- Final model saved to `./models/ppo_model_final.zip`

**This is your safety net.** If live training breaks on hackathon day, you have a pre-trained model to fall back on.

**Progress bar example:**
```
 50% ━━━━━━━━━━━━━━     |████████████                  | 25,000/50,000
```

---

**After 20 min:** You should see:
```
✓ Training complete. Model saved to ./models/ppo_model_final.zip
```

## Step 9: Test Evaluation (2 min)

Load the pre-trained model and run one episode:

```bash
python eval.py
```

**What happens:**
- Opens MuJoCo viewer (3D window)
- Shows robot moving around
- Takes ~30 seconds
- Prints final statistics

**Expected output:**
```
🤖 Running evaluation episode with trained model...

📊 Episode Results:
   Total Reward: -125.34
   Length: 150 steps
   Survivor Reached: False
```

*(Reaching the survivor on first try is unlikely without extensive training — that's OK.)*

---

## ✨ You're Done!

Your environment is ready. All team members should:

1. Run steps 1–9 above
2. Confirm `python train.py` works
3. Confirm `python eval.py` opens the viewer

**Before hackathon day:** Everyone should have `./models/ppo_model_final.zip` saved locally.

## 🆘 Troubleshooting

### "conda: command not found"
```bash
# Restart your terminal after installing miniforge
# Or manually initialize conda:
~/miniforge3/bin/conda init
source ~/.zshrc  # or ~/.bashrc
```

### "No module named 'mujoco'"
```bash
conda activate disaster
pip install mujoco
```

### "bracket error" in pip command
Use quotes:
```bash
pip install "gymnasium[mujoco]"  # NOT pip install gymnasium[mujoco]
```

### "MuJoCo viewer doesn't open"
- This is very rare on M-series Macs
- Try: `python -c "import mujoco.viewer; print('OK')"`
- If that hangs, try: `pip uninstall mujoco && pip install mujoco`

### Training is very slow
- Normal on CPU. ~2–3 min per 5k steps on M4 CPU.
- If it's taking >5 min per 5k steps, something may be wrong.
- Check: Are other apps hogging CPU? (Activity Monitor on Mac)

### "stable_baselines3 not compatible"
We pinned version 2.3.2 which works well with gymnasium. Don't upgrade:
```bash
pip install 'stable-baselines3==2.3.2'  # Use this version
```

---

## 📞 Questions?

- Stuck on step 1? Check miniforge docs: https://github.com/conda-forge/miniforge
- Stuck on MuJoCo? Check MuJoCo docs: https://mujoco.readthedocs.io/
- Stuck on stable-baselines3? Check SB3 docs: https://stable-baselines3.readthedocs.io/

**Next:** Read `README.md` for project overview. Good luck! 🚀


# Hackathon Day Timeline & Checklist

**Event:** Google I/O Hackathon  
**Location:** Shack15, Ferry Building, San Francisco  
**Date:** May 23, 2026  
**Time:** 10:30 AM – 5:00 PM

---

## 🎯 Goals for the Day

1. ✅ Integrate real Gemini 3.5 Flash API
2. ✅ Train on 200k steps (or use pre-trained fallback)
3. ✅ Build ScenarioAgent (scene generation)
4. ✅ Build DebriefAgent (trajectory analysis)
5. ✅ Wire FastAPI frontend
6. ✅ Demo end-to-end pipeline
7. ✅ Record 60-second video + submit

---

## 👥 Team Roles (3 People)

For a team of **3 people with 1 frontend person**, use this split:

| Person | Role | Owns |
|--------|------|------|
| **Person A (Frontend)** | UI / demo experience | `frontend/` — HTML, CSS, JS, trajectory viz, loading/errors, demo buttons |
| **Person B (RL)** | RL + simulation | `disaster_env.py`, `train.py`, `eval.py`, model, **`run_episode()`** function |
| **Person C (Agents/Backend)** | Agents + API integration | `scenario_agent.py`, `debrief_agent.py`, `mock_gemini.py`, **`app.py`**, video, submit |

**Person C is backend lead AND integration/demo lead.** Person B stays on physics + policy so training runs in background.

### How Work Flows (Kitchen Analogy)

- **Frontend** = dining room (what judges see)
- **Person C** = waiter + head chef (API + Gemini)
- **Person B** = specialist cook (robot actually moves)

Person B exposes one function Person C calls:

```python
# rescue_runner.py (Person B owns this)
def run_episode(scene: dict, model_path: str = "./models/ppo_model_final") -> dict:
    """Returns { trajectory: [[x,y],...], reached: bool, steps: int }"""
```

Person C's `/rescue` route:

```
disaster text → Gemini scene → run_episode() → Gemini debrief → JSON back to frontend
```

Frontend only ever calls `fetch('/rescue', ...)`.

### API Contract (Agree by 10:30)

Lock this interface so you can build in parallel:

```json
POST /rescue
{ "disaster_description": "A 7.5 magnitude earthquake..." }

→ {
  "scene": { 
    "robot_start": [x, y],
    "survivor_pos": [x, y],
    "obstacles": [{pos: [x,y], size: [w,h]}, ...],
    "hazards": [{center: [x,y], radius: r}, ...],
    "difficulty": "easy|medium|hard"
  },
  "trajectory": [[x, y], [x, y], ...],
  "reached": false,
  "steps": 150,
  "debrief": "The robot navigated through..."
}
```

**Person C ships a mock `/rescue`** that returns fixed JSON by 11:00 AM. Frontend wires the real API at 1:00 PM.

### Hour-by-Hour Breakdown

| Time | Frontend (Person A) | Person B (RL) | Person C (Agents/API) |
|------|-------------------|---------------|------------------------|
| **10:00** | UI against `mocks/rescue_response.json` | Verify pre-trained model, `eval.py` | Get API key, stub `app.py` + mock `/rescue` |
| **10:30** | Input form + loading spinner | Finish `DisasterEnv` tweaks if needed | Ship `scenario_agent.py`, test real Gemini |
| **11:00** | Trajectory map + debrief panel | **Start 200k training**, build `run_episode()` | `debrief_agent.py`, wire mock `/rescue` |
| **12:00** | CSS + error handling (empty input, long text) | Monitor training, **don't stop it** | Serve frontend static files from FastAPI |
| **1:00 PM** | Switch from mock → real `/rescue` API | Glance at training logs | Integration test full pipeline |
| **2:00 PM** | Polish UI only | Fix RL bugs if rescue looks wrong | Error handling + fallbacks |
| **3:00–4:00 PM** | Demo buttons + rehearsal | Training should finish or fallback to pre-trained | **Loom video + GitHub push + submit** |
| **4:30 PM** | ❌ Stop new features | ❌ Stop new features | ✅ Submit form |

### What You Should NOT Do

- **Frontend:** Don't build `app.py` or import MuJoCo/PPO/Gemini
- **Frontend:** Don't wait for backend before 11 AM — use mock JSON
- **Frontend:** Don't own integration testing at 1 PM — Person C runs that; you confirm the UI works
- **Person B:** Don't merge training changes after 1:30 PM — use pre-trained if slow
- **Person C:** Don't rebuild the UI — use what Frontend ships

### Emergency Fallbacks

| Problem | Who Handles | Backup |
|---------|-------------|--------|
| Training too slow | **Person B** — stop at 1:30 PM | Use `./models/ppo_model_final.zip` |
| Gemini API down | **Person C** — activate `mock_gemini.py` | Demo still works with hardcoded scenes |
| MuJoCo viewer broken | **Person B** — headless run | **Person A** shows trajectory on canvas |
| Demo crashes last-minute | **Person C** runs `python eval.py` | **Person A** keeps UI responsive |

---

## ⏰ Timeline

### 10:00 AM — Arrive, Setup

**All:** 
- [ ] Everyone activates `conda activate disaster`
- [ ] Pull latest code: `git pull origin main`

**Person B:**
- [ ] Verify pre-trained model exists: `ls -la models/ppo_model_final.zip`
- [ ] Quick test: `python eval.py` (should run in <1 min with no render)

**Person A (Frontend):**
- [ ] Verify `mocks/rescue_response.json` exists and has valid structure
- [ ] Open dev tools, test mock fetch locally

### 10:30 AM — Get API Keys & Integrate Gemini

**Person C (Agents/Backend):**  
**Duration:** 30 min

- [ ] Get Google Cloud API key from hackathon organizers
- [ ] Create `scenario_agent.py`:

```python
import google.generativeai as genai
import json

genai.configure(api_key="YOUR_API_KEY")

def get_scene_from_gemini(disaster_description: str) -> dict:
    """Call real Gemini 3.5 Flash to generate scene config."""
    
    prompt = f"""
Given this disaster description:
{disaster_description}

Return a JSON scene config with:
- robot_start: [x, y] starting position
- survivor_pos: [x, y] survivor position  
- obstacles: list of {{pos: [x, y], size: [w, h]}}
- hazards: list of {{center: [x, y], radius: r}}
- difficulty: "easy", "medium", or "hard"

Return ONLY valid JSON, no other text.
"""
    
    model = genai.GenerativeModel("gemini-3.5-flash")
    response = model.generate_content(prompt)
    
    try:
        config = json.loads(response.text)
        return config
    except json.JSONDecodeError:
        # Fallback if Gemini returns invalid JSON
        from mock_gemini import get_scene_config
        return get_scene_config(disaster_description)
```

- [ ] Test it: `python -c "from scenario_agent import get_scene_from_gemini; print(get_scene_from_gemini('earthquake'))"`
- [ ] Should print valid JSON within 5 sec

**Fallback:** If API is slow or fails, the code falls back to `mock_gemini.py` automatically.

### 11:00 AM — Start Training, Parallel Work Begins

**THREE PARALLEL TRACKS:**

---

**Person B (RL Training):**
```bash
# Start 200k step training (runs for ~45 min in background)
python train.py  # (modify to 200k steps instead of 50k)
# Monitor but DON'T STOP unless at 1:30 PM it's not done
```

Also build `rescue_runner.py`:
```python
# rescue_runner.py
def run_episode(scene: dict, model_path: str = "./models/ppo_model_final") -> dict:
    """Run one episode with given scene config."""
    from disaster_env import DisasterEnv
    from stable_baselines3 import PPO
    
    env = DisasterEnv()
    obs, _ = env.reset()
    
    trajectory = []
    model = PPO.load(model_path)
    
    for step in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        trajectory.append(list(obs[:2]))
        
        if terminated or truncated:
            break
    
    env.close()
    return {
        "trajectory": trajectory,
        "reached": info.get("reached", False),
        "steps": len(trajectory)
    }
```

---

**Person C (Agents/Backend):**
Build `debrief_agent.py`:

```python
import google.generativeai as genai
import json

def analyze_trajectory(robot_path: list, steps: int, reached: bool) -> str:
    """
    Given robot path and episode stats, have Gemini summarize the rescue attempt.
    
    Args:
        robot_path: list of [x, y] positions over time
        steps: number of steps taken
        reached: whether survivor was reached
    
    Returns:
        Natural language debrief string
    """
    
    prompt = f"""
A rescue robot attempted to reach a survivor in a disaster zone.

Movement log:
- Steps taken: {steps}
- Path length: {len(robot_path)} waypoints
- Survivor reached: {reached}

If the path is available, summarize the robot's movement and rescue success briefly in 2-3 sentences.
Keep it conversational and highlight any challenges overcome.
"""
    
    genai.configure(api_key="YOUR_API_KEY")
    model = genai.GenerativeModel("gemini-3.5-flash")
    response = model.generate_content(prompt)
    
    return response.text
```

Then build `app.py` to wire everything:

```python
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from scenario_agent import get_scene_from_gemini
from debrief_agent import analyze_trajectory
from rescue_runner import run_episode
import json

app = FastAPI()
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Load trained model path (Person B updates this)
MODEL_PATH = "./models/ppo_model_final"

try:
    from stable_baselines3 import PPO
    PPO.load(MODEL_PATH)
    print("✓ Model loaded")
except:
    print("⚠️ Model not found, will fallback at runtime")

@app.get("/")
async def root():
    with open("frontend/index.html") as f:
        return HTMLResponse(f.read())

@app.post("/rescue")
async def rescue(request: dict):
    # Get scene from Gemini
    scene = get_scene_from_gemini(request["disaster_description"])
    
    # Run episode with Person B's function
    result = run_episode(scene, MODEL_PATH)
    
    # Get debrief from Gemini
    debrief = analyze_trajectory(
        result["trajectory"],
        result["steps"],
        result["reached"]
    )
    
    return {
        "scene": scene,
        "trajectory": result["trajectory"],
        "reached": result["reached"],
        "steps": result["steps"],
        "debrief": debrief
    }
```

Test mock endpoint:
```bash
python -m uvicorn app:app --reload
# Visit http://localhost:8000
```

---

**Person A (Frontend):**
Start building `frontend/index.html` with mock fetch:

```html
<!-- frontend/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🤖 Disaster Rescue Robot</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 700px;
            width: 100%;
            padding: 40px;
        }
        h1 {
            color: #333;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        .subtitle { color: #666; margin-bottom: 30px; }
        .input-group {
            display: flex;
            gap: 10px;
            margin-bottom: 30px;
        }
        input {
            flex: 1;
            padding: 12px 16px;
            font-size: 16px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            transition: border-color 0.3s;
        }
        input:focus {
            outline: none;
            border-color: #667eea;
        }
        button {
            padding: 12px 28px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.3s;
        }
        button:hover { background: #5568d3; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        .loading { display: none; text-align: center; margin: 20px 0; }
        .spinner {
            display: inline-block;
            width: 40px;
            height: 40px;
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .result { display: none; margin-top: 30px; }
        .section {
            margin-bottom: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        h3 { color: #333; margin-bottom: 10px; font-size: 1.2em; }
        .scene-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            font-size: 14px;
        }
        .scene-item {
            padding: 10px;
            background: white;
            border-radius: 4px;
            border: 1px solid #e0e0e0;
        }
        .scene-label { font-weight: 600; color: #667eea; }
        .trajectory-canvas {
            width: 100%;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            background: white;
            margin-top: 10px;
        }
        .debrief { font-size: 15px; line-height: 1.6; color: #333; }
        .stats { font-size: 14px; color: #666; margin-top: 10px; }
        .error { color: #d32f2f; background: #ffebee; padding: 15px; border-radius: 8px; display: none; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Disaster Rescue Robot</h1>
        <p class="subtitle">Powered by Gemini + Reinforcement Learning</p>
        
        <div class="input-group">
            <input type="text" id="disaster" placeholder="Describe the disaster scenario..."
                   value="A 7.5 magnitude earthquake has devastated downtown...">
            <button onclick="launchRescue()" id="rescueBtn">Launch Rescue</button>
        </div>
        
        <div class="error" id="errorDiv"></div>
        <div class="loading" id="loading"><div class="spinner"></div><p>Launching rescue mission...</p></div>
        <div class="result" id="result">
            <div class="section">
                <h3>📍 Scene Configuration</h3>
                <div class="scene-grid" id="sceneGrid"></div>
            </div>
            <div class="section">
                <h3>🗺️ Robot Trajectory</h3>
                <canvas id="trajectoryCanvas" class="trajectory-canvas" width="400" height="300"></canvas>
            </div>
            <div class="section">
                <h3>📊 Mission Debrief</h3>
                <p class="debrief" id="debrief"></p>
                <div class="stats" id="stats"></div>
            </div>
        </div>
    </div>

    <script>
        async function launchRescue() {
            const disaster = document.getElementById('disaster').value.trim();
            if (!disaster) {
                showError('Please describe a disaster scenario.');
                return;
            }
            
            document.getElementById('rescueBtn').disabled = true;
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            document.getElementById('errorDiv').style.display = 'none';
            
            try {
                const response = await fetch('/rescue', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({disaster_description: disaster})
                });
                
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                const data = await response.json();
                
                displayResult(data);
            } catch (err) {
                showError(`Error: ${err.message}`);
            } finally {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('rescueBtn').disabled = false;
            }
        }
        
        function displayResult(data) {
            // Scene config
            const sceneHtml = `
                <div class="scene-item">
                    <div class="scene-label">Start</div>
                    [${data.scene.robot_start.map(x => x.toFixed(1)).join(', ')}]
                </div>
                <div class="scene-item">
                    <div class="scene-label">Survivor</div>
                    [${data.scene.survivor_pos.map(x => x.toFixed(1)).join(', ')}]
                </div>
                <div class="scene-item">
                    <div class="scene-label">Difficulty</div>
                    ${data.scene.difficulty}
                </div>
                <div class="scene-item">
                    <div class="scene-label">Obstacles</div>
                    ${data.scene.obstacles?.length || 0}
                </div>
            `;
            document.getElementById('sceneGrid').innerHTML = sceneHtml;
            
            // Trajectory canvas
            drawTrajectory(data.trajectory, data.scene);
            
            // Debrief + stats
            document.getElementById('debrief').textContent = data.debrief;
            document.getElementById('stats').innerHTML = `
                <strong>Steps:</strong> ${data.steps} | 
                <strong>Survivor Reached:</strong> ${data.reached ? '✓ Yes' : '✗ No'}
            `;
            
            document.getElementById('result').style.display = 'block';
        }
        
        function drawTrajectory(trajectory, scene) {
            const canvas = document.getElementById('trajectoryCanvas');
            const ctx = canvas.getContext('2d');
            
            // Clear and draw background
            ctx.fillStyle = '#ffffff';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // Draw grid
            ctx.strokeStyle = '#f0f0f0';
            ctx.lineWidth = 1;
            for (let i = 0; i <= 10; i++) {
                ctx.beginPath();
                ctx.moveTo((canvas.width / 10) * i, 0);
                ctx.lineTo((canvas.width / 10) * i, canvas.height);
                ctx.stroke();
                ctx.beginPath();
                ctx.moveTo(0, (canvas.height / 10) * i);
                ctx.lineTo(canvas.width, (canvas.height / 10) * i);
                ctx.stroke();
            }
            
            if (!trajectory || trajectory.length === 0) return;
            
            const scale = 40; // pixels per unit
            const offsetX = canvas.width / 2;
            const offsetY = canvas.height / 2;
            
            // Draw trajectory line
            ctx.strokeStyle = '#667eea';
            ctx.lineWidth = 2;
            ctx.beginPath();
            trajectory.forEach((pt, i) => {
                const x = offsetX + pt[0] * scale;
                const y = offsetY - pt[1] * scale;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.stroke();
            
            // Draw start and end
            const startX = offsetX + trajectory[0][0] * scale;
            const startY = offsetY - trajectory[0][1] * scale;
            const endX = offsetX + trajectory[trajectory.length-1][0] * scale;
            const endY = offsetY - trajectory[trajectory.length-1][1] * scale;
            
            ctx.fillStyle = '#4caf50';
            ctx.beginPath();
            ctx.arc(startX, startY, 6, 0, Math.PI * 2);
            ctx.fill();
            
            ctx.fillStyle = '#ff9800';
            ctx.beginPath();
            ctx.arc(endX, endY, 6, 0, Math.PI * 2);
            ctx.fill();
            
            // Legend
            ctx.font = '12px Arial';
            ctx.fillStyle = '#666';
            ctx.fillText('● Start', 10, canvas.height - 10);
            ctx.fillStyle = '#ff9800';
            ctx.fillText('● End', canvas.width - 80, canvas.height - 10);
        }
        
        function showError(msg) {
            document.getElementById('errorDiv').textContent = msg;
            document.getElementById('errorDiv').style.display = 'block';
        }
        
        // Allow Enter key to launch
        document.getElementById('disaster').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') launchRescue();
        });
    </script>
</body>
</html>
```

Save to: `frontend/index.html`

- [ ] Test with mock: Create `mocks/rescue_response.json` and test locally
- [ ] Update fetch URL to `http://localhost:8000/rescue` once Person C ships the mock endpoint

### 12:00 PM — Lunch + Check Training

**Person B (RL):**
- [ ] Training should be ~25% done (50k/200k)
- [ ] Glance at logs — is it progressing or stuck?
- [ ] If fine: **keep it running, don't touch it**
- [ ] If slow: note the time, might need fallback at 1:30 PM

**Person C (Agents/Backend):**
- [ ] Wire mock `/rescue` endpoint so Person A can test real UI at 1:00 PM
- [ ] Verify `scenario_agent.py` and `debrief_agent.py` work with real Gemini API

**Person A (Frontend):**
- [ ] Polish CSS and error handling
- [ ] Test empty inputs, long inputs, special characters
- [ ] Verify trajectory canvas renders correctly with mock data

### 1:00 PM — Integration Testing

**Person C (Agents/Backend) — LEAD THIS:**
- [ ] Test API key works: `python -c "from scenario_agent import get_scene_from_gemini; print(get_scene_from_gemini('test'))"`
- [ ] Test full `/rescue` endpoint with real Gemini
- [ ] Wire `run_episode()` from Person B into the route
- [ ] Serve frontend static files from FastAPI

**Person A (Frontend) — VERIFY:**
- [ ] User can type disaster description
- [ ] Submit button triggers fetch
- [ ] Real JSON appears (no longer mocked)
- [ ] Trajectory visualizes correctly
- [ ] Debrief text displays

**Person B (RL) — CHECK:**
- [ ] Training still running? How many steps completed?
- [ ] If >150k: training is on track
- [ ] If <100k: may need fallback at 1:30 PM

### 2:00 PM — Debug & Polish (No new features after this)

**Person C (Agents/Backend):**
- [ ] Fix any API errors or timeouts
- [ ] Add fallback to `mock_gemini.py` if Gemini is slow
- [ ] Ensure `/rescue` returns valid JSON always

**Person A (Frontend):**
- [ ] Fix layout issues on mobile/tablet
- [ ] Add error message display if `/rescue` fails
- [ ] Polish loading spinner and transitions
- [ ] Test 3 disaster scenarios end-to-end

**Person B (RL):**
- [ ] Check training status one more time
- [ ] If training finished: great! New model is ready
- [ ] If training stalled: **stop it at 2:30 PM and use pre-trained `ppo_model_final.zip`**

### 3:00 PM — Demo Rehearsal (Stop building at 3:30)

**Person C (Agents/Backend) + Person A (Frontend) — DEMO LEAD:**
- [ ] Prepare 2 hardcoded disaster scenarios:
  ```
  Scenario 1: "A 7.5 magnitude earthquake has devastated downtown. 
              Multiple buildings collapsed. Survivor 50 meters northeast."
  
  Scenario 2: "A flood has swept through the city. Survivor trapped 
              on the 3rd floor of a submerged building."
  ```

- [ ] Run full demo 3 times, timing: <3 min per run
- [ ] Practice talking points:
  - "Gemini generates a dynamic scene with obstacles"
  - "Our RL policy navigates the robot to the survivor"
  - "Gemini analyzes the trajectory and provides a debrief"

**Person B (RL):**
- [ ] Have `eval.py` ready as a backup demo (shows just the RL)
- [ ] Test it locally: `python eval.py` (should run in <30 sec)

### 4:00 PM — Record Video & Submit

**Person C (Agents/Backend) — LEAD THIS:**

**Record a 60-second Loom:**
1. Open `http://localhost:8000`
2. Type disaster: *"A 7.5 magnitude earthquake has devastated downtown..."*
3. Show loading spinner
4. Show Gemini-generated scene JSON
5. Show robot trajectory canvas
6. Show final debrief text
7. Do one clean run (60 sec total)
8. Upload to Loom.com (free)
9. Get shareable link

**Push to GitHub:**
```bash
git add -A
git commit -m "Final demo: disaster rescue robot with Gemini + RL"
git push origin main
```

**Prepare submission:**
- [ ] All 3 people listed as contributors
- [ ] README has: project description + setup instructions
- [ ] `plan.md` + API contract documented in repo
- [ ] Loom video link ready

**Submit form:** https://cerebralvalley.ai/e/google-io-hackathon/hackathon/submit
- [ ] Project name: "Disaster Rescue Robot"
- [ ] Team members: All 3 names
- [ ] Repo link (GitHub public)
- [ ] Loom video link
- [ ] Description: "AI system combining Gemini for scene generation, RL policy for navigation, and Gemini for trajectory analysis. Live demo generates disaster scenarios and rescues survivors."

**Deadline: 5:00 PM**

### 4:30 PM — Stop Building

**All teams: No new features.** Only submit and prepare for live demo.

- [ ] All code committed and pushed
- [ ] Everyone rehearsed the 3-minute pitch
- [ ] Backup: `python eval.py` ready if web demo crashes

### 5:00 PM — Submit & Celebrate 🎉

---

## 🚨 Emergency Fallbacks

### Training Takes Too Long
- **If training > 1:30 PM:** Stop it and use `./models/ppo_model_final.zip` from pre-hackathon
- You still get full credit — the policy works, just slightly less trained

### Gemini API is Down
- Code automatically falls back to `mock_gemini.py`
- Demo still works, just with hardcoded scenes
- Explain to judges: "Real Gemini had network issues, but the pipeline is complete"

### Viewer Doesn't Open
- Run MuJoCo headless (no render)
- Just show the debrief stats + trajectory data
- Still a cool demo!

### Something Breaks Last Minute
- You have `./models/ppo_model_final.zip` trained beforehand
- You can demo just the RL policy: `python eval.py`
- Judges will still see a functioning AI system

---

## 📋 Final Checklist (4:55 PM)

**All three people:**
- [ ] Repo pushed to GitHub (public)
- [ ] All team members' names in README + repo description
- [ ] README has: project overview, setup instructions, API docs
- [ ] Video recorded and uploaded (Loom link in submission)
- [ ] Form submitted with all correct links
- [ ] Everyone rehearsed the 3-minute pitch
- [ ] Demo tested once live — no crashes

**Person B (RL):**
- [ ] `eval.py` works as backup demo
- [ ] Pre-trained model is in `./models/ppo_model_final.zip`

**Person C (Agents/Backend):**
- [ ] `/rescue` endpoint responds in <5 sec
- [ ] Fallback to `mock_gemini.py` if API is slow
- [ ] Loom video link in submission form

**Person A (Frontend):**
- [ ] UI renders on judges' screen resolution
- [ ] Loading spinner visible during API call
- [ ] Error messages are clear

---

## 🎤 Live Demo Talking Points (3 min)

**Intro (30 sec):**
"We built an AI system that rescues survivors in disaster scenarios. It combines three AI components: Gemini generates dynamic environments, our RL policy navigates the robot, and Gemini provides real-time analysis."

**Demo (2 min):**
1. Type disaster: *"7.5 magnitude earthquake, multiple collapsed buildings, survivor northeast"*
2. Show scene JSON: *"Real Gemini generates obstacles and hazard zones dynamically"*
3. Run episode: *"Our trained RL policy navigates to reach the survivor"*
4. Show trajectory map: *"Robot's actual path through the environment"*
5. Show debrief: *"Gemini analyzes the rescue and summarizes what happened"*

**Close (30 sec):**
"The system handles scenario generation, policy learning, and analysis — three specialized AI models working in concert. This could scale to real disaster response with multi-survivor coordination."

---

**Division of Speaking:**
- **Person A:** Intro + UI/demo walkthrough ("Type in the disaster...")
- **Person C:** Scene generation + Gemini explanations ("Gemini generates...")
- **Person B:** RL policy explanation ("Our trained policy navigates...")

---

## 🏆 What Judges Are Looking For

- **Impact Potential (20%):** Can this scale? Real-world rescue ops?
- **Live Demo (45%):** Does it work smoothly right now?
- **Creativity (35%):** Is this novel? Smart use of Gemini?

**Our advantages:**
- ✅ Novel: RL + generative AI + multi-agent
- ✅ Uses Gemini for 2+ things (scenario + debrief)
- ✅ Live demo is visual and exciting
- ✅ Shows real technical depth

---

## 📞 Emergency Contacts

- **API Issues:** Google I/O support desk @ venue
- **Technical Questions:** Slack #google-deepmind channel
- **General Hackathon Q's:** Slack #questions, ping @CV

---

**Good luck! You've got this. 🚀**