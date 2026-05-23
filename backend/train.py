"""
Train a PPO policy on the 3-D DisasterEnv.
Run:  uv run python train.py
"""

import os
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from rich.console import Console

from export_run import export_run

from disaster_env import DisasterEnv
from scenes import GENERATED_SCENES, get_scene

console = Console()
MODELS_DIR    = "./models"
TOTAL_STEPS   = 500_000
N_ENVS        = len(GENERATED_SCENES)
CHECKPOINT_FREQ = 20_000   # save every N steps (per env)
MODEL_NAME    = "ppo_buried_detection"


def make_env(scene_index: int):
    def _init():
        return DisasterEnv(scene=get_scene(scene_index), render_mode="rgb_array")
    return _init


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    console.print("[bold cyan]Building vectorised environment from fixed generated scenes...[/]")
    for idx, scene in enumerate(GENERATED_SCENES):
        console.print(f"  {idx + 1}. {scene['name']}")

    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    env = VecMonitor(env)

    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=MODELS_DIR,
        name_prefix=MODEL_NAME,
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
        ent_coef=0.0,
        tensorboard_log="./tb_logs",
    )

    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=checkpoint_cb,
        progress_bar=True,
    )

    final_path = os.path.join(MODELS_DIR, MODEL_NAME + "_final")
    model.save(final_path)
    console.print(f"[bold green]✓ Training complete. Model saved to {final_path}.zip[/]")

    compatibility_path = Path(MODELS_DIR) / "ppo_model_final"
    model.save(compatibility_path)
    console.print(f"[green]Also wrote compatibility model to {compatibility_path}.zip[/]")
    env.close()

    console.print("[bold cyan]Exporting run manifest to runs/…[/]")
    export_run(
        final_path,
        run_name=f"{MODEL_NAME}_final",
        total_steps=TOTAL_STEPS,
        n_envs=N_ENVS,
    )


if __name__ == "__main__":
    main()
