"""
train_from_run.py — Behavior Cloning (BC) warm-start pretraining from a prior run's
successful trajectories, followed by normal PPO learning.

Usage:
    uv run python train_from_run.py --prior-run ppo_buried_detection_final --bc-epochs 5 --total-steps 200000
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from disaster_env import DisasterEnv
from episode_loader import load_transitions
from export_run import export_run
from run_store import run_dir, read_json
from scenes import GENERATED_SCENES, get_scene


def make_env(scene_index: int):
    def _init():
        return DisasterEnv(scene=get_scene(scene_index), render_mode="rgb_array")
    return _init


def main():
    parser = argparse.ArgumentParser(description="BC pretrain warm-start + PPO learn.")
    parser.add_argument("--prior-run", required=True, help="run_id of the prior run")
    parser.add_argument("--bc-epochs", type=int, default=5, help="Number of BC epochs")
    parser.add_argument("--total-steps", type=int, default=200000, help="Total PPO training steps after BC")
    args = parser.parse_args()

    prior_run = args.prior_run
    bc_epochs = args.bc_epochs
    total_steps = args.total_steps

    # 1. Load transitions from the prior run's reached episodes
    print(f"Loading reached-episode transitions for prior run: {prior_run}…")
    transitions = load_transitions(prior_run)
    if not transitions:
        print(f"Error: No successful transitions with rollouts found in runs/{prior_run}/episodes/")
        sys.exit(1)
    print(f"Loaded {len(transitions)} transitions.")

    # 2. Locate the prior model checkpoint path
    # Look up in runs/<prior_run>/summary.json
    summary_path = run_dir(prior_run) / "summary.json"
    model_path = None
    if summary_path.exists():
        try:
            summary_data = read_json(summary_path)
            model_path_str = summary_data.get("model_path")
            if model_path_str:
                potential_path = _HERE / model_path_str
                if potential_path.exists():
                    model_path = potential_path
        except Exception:
            pass

    if model_path is None:
        # Fallback to models/<prior_run>.zip or models/<prior_run>
        for p in [
            _HERE / "models" / f"{prior_run}.zip",
            _HERE / "models" / prior_run,
            _HERE / "models" / "ppo_buried_detection_final.zip",
        ]:
            if p.exists():
                model_path = p
                break

    if model_path is None:
        print(f"Error: Prior checkpoint model zip not found for run {prior_run}.")
        sys.exit(1)

    print(f"Loading checkpoint: {model_path}…")

    # 3. Build vectorized environment (needed for checking compatibility & training)
    print("Building vectorized environment…")
    N_ENVS = len(GENERATED_SCENES)
    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    env = VecMonitor(env)

    # 4. Load the prior policy & verify compatibility
    try:
        # Load model structure without env first to compare shapes
        model = PPO.load(model_path, device="cpu")
        if model.observation_space.shape != env.observation_space.shape:
            env.close()
            print(
                f"Error: Prior policy observation space {model.observation_space.shape} "
                f"is incompatible with current environment observation space {env.observation_space.shape}."
            )
            sys.exit(1)
            
        # Re-load with the environment
        model = PPO.load(model_path, env=env, device="cpu")
    except Exception as e:
        env.close()
        print(f"Error loading prior model structure: {e}")
        sys.exit(1)

    # 5. Behavior Cloning pretraining
    obs_all = np.array([t[0] for t in transitions], dtype=np.float32)
    act_all = np.array([t[1] for t in transitions], dtype=np.float32)

    obs_tensor = torch.as_tensor(obs_all, device=model.policy.device)
    act_tensor = torch.as_tensor(act_all, device=model.policy.device)

    dataset = torch.utils.data.TensorDataset(obs_tensor, act_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    optimizer = optim.Adam(model.policy.parameters(), lr=3e-4)
    loss_fn = nn.MSELoss()

    print(f"Pretraining policy parameters using BC on device: {model.policy.device}…")
    for epoch in range(bc_epochs):
        epoch_loss = 0.0
        for batch_obs, batch_act in loader:
            optimizer.zero_grad()
            # Predict the actions distribution mean/mode
            dist = model.policy.get_distribution(batch_obs)
            pred_actions = dist.mode()
            loss = loss_fn(pred_actions, batch_act)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_obs)
        mean_loss = epoch_loss / len(transitions)
        print(f"  BC Epoch {epoch+1}/{bc_epochs} - Loss: {mean_loss:.6f}")

    # 6. Normal PPO reinforcement learning step
    MODELS_DIR = _HERE / "models"
    os.makedirs(MODELS_DIR, exist_ok=True)
    CHECKPOINT_FREQ = 20_000

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bc_model_name = f"ppo_buried_detection_bc_{timestamp}"

    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=str(MODELS_DIR),
        name_prefix=bc_model_name,
        verbose=1,
    )

    print(f"Starting normal reinforcement learning for {total_steps:,} steps…")
    model.learn(
        total_timesteps=total_steps,
        callback=checkpoint_cb,
        progress_bar=True,
    )

    final_path = MODELS_DIR / bc_model_name
    model.save(final_path)
    print(f"✓ Saved BC PPO model to {final_path}.zip")

    compatibility_path = MODELS_DIR / "ppo_model_final"
    model.save(compatibility_path)
    print(f"✓ Also wrote compatibility model to {compatibility_path}.zip")
    env.close()

    # 7. Export the run
    print("Exporting training run…")
    export_run(
        final_path,
        run_name=bc_model_name,
        total_steps=total_steps,
        n_envs=N_ENVS,
        init_from=f"runs/{prior_run}",
        bc_transitions=len(transitions),
        bc_epochs=bc_epochs,
    )


if __name__ == "__main__":
    main()
