"""
Train a PPO policy on the 3-D DisasterEnv.
Run:  uv run python train.py
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from rich.console import Console

from disaster_env import DisasterEnv

console = Console()
MODELS_DIR    = "./models"
TOTAL_STEPS   = 200_000
N_ENVS        = 6
CHECKPOINT_FREQ = 20_000   # save every N steps (per env)


def make_env():
    def _init():
        return DisasterEnv(render_mode="rgb_array")
    return _init


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    console.print("[bold cyan]Building vectorised environment...[/]")
    env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])
    env = VecMonitor(env)

    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=MODELS_DIR,
        name_prefix="ppo_disaster",
        verbose=1,
    )

    console.print(f"[bold green]Training PPO for {TOTAL_STEPS:,} steps across {N_ENVS} envs[/]")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="cpu",
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log="./tb_logs",
    )

    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=checkpoint_cb,
        progress_bar=True,
    )

    final_path = os.path.join(MODELS_DIR, "ppo_model_final")
    model.save(final_path)
    console.print(f"[bold green]✓ Training complete. Model saved to {final_path}.zip[/]")
    env.close()


if __name__ == "__main__":
    main()
