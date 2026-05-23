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

## ⏰ Timeline

### 10:00 AM — Arrive, Setup

- [ ] Everyone activates `conda activate disaster`
- [ ] Pull latest code: `git pull origin main`
- [ ] Verify pre-trained model exists: `ls -la models/ppo_model_final.zip`
- [ ] Quick test: `python eval.py` (should run in <1 min with no render)

### 10:30 AM — Get API Keys & Integrate Gemini

**Who:** Agent Lead  
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
    
    model = genai.GenerativeModel("gemini-1.5-flash")
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

**Who:** RL Lead starts training; Agents + Frontend teams work in parallel

**RL Lead:**
```bash
# Start 200k step training (runs for ~45 min)
python train.py  # (modify to 200k steps instead of 50k)
```

**Agent Team:**
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
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    
    return response.text
```

**Frontend Team:**
Start building `app.py`:

```python
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from scenario_agent import get_scene_from_gemini
from debrief_agent import analyze_trajectory
from disaster_env import DisasterEnv
from stable_baselines3 import PPO
import json

app = FastAPI()

# Load trained model
try:
    model = PPO.load("./models/ppo_model_final")
except:
    print("⚠️  Model not found, will use random policy")
    model = None

@app.get("/")
async def root():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Disaster Rescue</title>
        <style>
            body { font-family: Arial; max-width: 800px; margin: 0 auto; padding: 20px; }
            input { width: 100%; padding: 10px; font-size: 16px; }
            button { padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; }
            #result { margin-top: 20px; padding: 15px; background: #f0f0f0; border-radius: 5px; }
            pre { background: #fff; padding: 10px; overflow-x: auto; }
        </style>
    </head>
    <body>
        <h1>🤖 Disaster Rescue Robot</h1>
        <input type="text" id="disaster" placeholder="Describe the disaster...">
        <button onclick="rescue()">Launch Rescue</button>
        <div id="result"></div>
        
        <script>
            async function rescue() {
                const disaster = document.getElementById('disaster').value;
                const response = await fetch('/rescue', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({disaster_description: disaster})
                });
                const data = await response.json();
                document.getElementById('result').innerHTML = `
                    <h3>Scene Config</h3>
                    <pre>${JSON.stringify(data.scene, null, 2)}</pre>
                    <h3>Debrief</h3>
                    <p>${data.debrief}</p>
                `;
            }
        </script>
    </body>
    </html>
    """)

@app.post("/rescue")
async def rescue(request: dict):
    # Get scene from Gemini
    scene = get_scene_from_gemini(request["disaster_description"])
    
    # Run episode
    env = DisasterEnv()
    obs, _ = env.reset()
    
    trajectory = []
    for _ in range(500):
        if model:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        trajectory.append(list(obs[:2]))  # robot position
        
        if terminated or truncated:
            break
    
    env.close()
    
    # Get debrief from Gemini
    reached = info.get("reached", False)
    debrief = analyze_trajectory(trajectory, len(trajectory), reached)
    
    return {
        "scene": scene,
        "trajectory_length": len(trajectory),
        "reached": reached,
        "debrief": debrief
    }
```

- [ ] Get FastAPI running: `python -m uvicorn app:app --reload`
- [ ] Test: Visit `http://localhost:8000`

### 12:00 PM — Lunch + Check Training

- [ ] Training should be halfway done (100k steps)
- [ ] Check TensorBoard: `tensorboard --logdir ./logs` (if available)
- [ ] If training is fine, keep it running
- [ ] If training is crashing, **use pre-trained model** and skip to step 13

### 1:00 PM — Integration Testing

**Who:** Demo/Integration Lead

- [ ] Test API key works
- [ ] Test scenario generation: `python -c "from scenario_agent import get_scene_from_gemini; print(get_scene_from_gemini('test'))"`
- [ ] Test debrief generation
- [ ] Test full loop: disaster description → scene → robot run → debrief
- [ ] Test FastAPI + browser UI

**Checklist:**
- [ ] User can type disaster description
- [ ] Scene JSON appears
- [ ] Robot runs (even if MuJoCo viewer is headless)
- [ ] Debrief text appears

### 2:00 PM — Debug & Polish

- [ ] Fix any broken components
- [ ] Test edge cases (empty input, very long input)
- [ ] Add error handling + fallbacks
- [ ] Make UI prettier (CSS, layout)

**Critical:** If training finished, celebrate! If not, that's OK — pre-trained model works.

### 3:00 PM — Demo Rehearsal

- [ ] Prepare 2 hardcoded disaster scenarios:
  ```
  Scenario 1: "A 7.5 magnitude earthquake has devastated downtown. Multiple buildings collapsed. We have a signal from a survivor 50 meters northeast."
  
  Scenario 2: "A flood has swept through the city. We have reports of a survivor trapped on the 3rd floor of a submerged building."
  ```

- [ ] Run full demo 3 times, timing it to <3 min per run
- [ ] Practice explaining what's happening:
  - "We input a disaster description"
  - "Gemini generates a dynamic scene with obstacles and survivor position"
  - "Our trained robot policy navigates to the survivor"
  - "Gemini analyzes the rescue attempt and provides a debrief"

### 4:00 PM — Record Video & Submit

**Who:** Demo Lead

**Record a 60-second Loom:**
1. Screen record your demo
2. Show: text input → Gemini scene → MuJoCo window → debrief output
3. Do one clean run (1 min)
4. Upload to Loom (free)
5. Get shareable link

**Prepare Submission:**
- [ ] Repo is public on GitHub
- [ ] All files committed and pushed
- [ ] README + SETUP + this file in repo
- [ ] One clear "Getting Started" section at top of README
- [ ] Loom video link ready

**Submit form at:** https://cerebralvalley.ai/e/google-io-hackathon/hackathon/submit
- [ ] Project name: "Disaster Rescue Robot"
- [ ] Team members: All names
- [ ] Repo link (GitHub public)
- [ ] Loom video link
- [ ] 1-2 sentence description

**Deadline: 5:00 PM**

### 4:30 PM — Stop Building

**No new features after 4:30 PM.** Polish and submit only.

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

- [ ] Repo pushed to GitHub (public)
- [ ] All team members' names in repo
- [ ] README has clear setup + project description
- [ ] Video recorded and uploaded (Loom link)
- [ ] Form submitted with correct links
- [ ] Demo tested at least once with no crashes
- [ ] Everyone knows what to say during live judging

---

## 🎤 Live Demo Talking Points (3 min)

**Intro (30 sec):**
"We built an AI-powered rescue robot using Gemini 3.5 Flash and reinforcement learning. Given a disaster scenario, our system generates a dynamic environment, trains a policy to navigate and rescue survivors."

**Demo (2 min):**
1. Type disaster: *"Earthquake in downtown area, survivor 5km northeast"*
2. Show scene JSON: *"Gemini generates obstacles and hazard zones"*
3. Run evaluation: *"Our trained RL policy navigates to the survivor"*
4. Show debrief: *"Gemini analyzes the trajectory and gives a natural language summary"*

**Close (30 sec):**
"The system combines scenario generation, RL policy learning, and trajectory analysis — three agents working together. Future work: real-time trajectory optimization, multi-survivor rescue."

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