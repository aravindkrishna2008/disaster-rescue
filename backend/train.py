"""
Train a PPO policy for Unitree G1 locomotion.

Examples:
    uv run python train.py --smoke --no-progress-bar
    uv run python train.py --curriculum --no-progress-bar
    uv run python train.py --timesteps 1000000
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from rich.console import Console
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from disaster_env import DEFAULT_BALANCE_ASSIST_SCALE, DisasterEnv
from scenes import GENERATED_SCENES, get_scene

console = Console()
MODELS_DIR = "./models"
DEFAULT_MODEL_NAME = "g1_locomotion_walk"
NATURAL_MODEL_NAME = "g1_locomotion_natural"
DEFAULT_TOTAL_STEPS = 1_000_000
DEFAULT_CHECKPOINT_FREQ = 50_000
STANDARD_CURRICULUM = (
    ("stand", 300_000, 1.0, 1.0),
    ("walk", 700_000, 1.0, 1.0),
    ("target", 1_000_000, 1.0, 1.0),
)
NATURAL_CURRICULUM = (
    ("stand", 200_000, 0.0, 0.0),
    ("flat_walk", 400_000, 0.95, DEFAULT_BALANCE_ASSIST_SCALE),
    ("low_assist_walk", 400_000, 0.95, DEFAULT_BALANCE_ASSIST_SCALE),
    ("obstacle_nav", 700_000, 0.95, DEFAULT_BALANCE_ASSIST_SCALE),
    ("natural_target", 900_000, 0.95, DEFAULT_BALANCE_ASSIST_SCALE),
)
CURRICULUM_STAGES = (
    "stand",
    "walk",
    "target",
    "flat_walk",
    "low_assist_walk",
    "obstacle_nav",
    "natural_target",
)


def make_env(scene_index: int, stage: str, assist_scale: float, balance_assist_scale: float):
    def _init():
        return DisasterEnv(
            scene=get_scene(scene_index),
            render_mode="rgb_array",
            curriculum_stage=stage,
            assist_scale=assist_scale,
            balance_assist_scale=balance_assist_scale,
        )

    return _init


def build_env(
    n_envs: int,
    use_subproc: bool,
    stage: str,
    assist_scale: float,
    balance_assist_scale: float,
):
    env_fns = [make_env(i, stage, assist_scale, balance_assist_scale) for i in range(n_envs)]
    if use_subproc and n_envs > 1:
        return VecMonitor(SubprocVecEnv(env_fns))
    return VecMonitor(DummyVecEnv(env_fns))


def parse_curriculum_steps(raw: str, expected_count: int) -> tuple[int, ...]:
    parts = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if len(parts) != expected_count:
        raise argparse.ArgumentTypeError(
            f"expected {expected_count} comma-separated step counts"
        )
    return tuple(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the G1 locomotion PPO policy.")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TOTAL_STEPS)
    parser.add_argument("--n-envs", type=int, default=len(GENERATED_SCENES))
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--checkpoint-freq", type=int, default=DEFAULT_CHECKPOINT_FREQ)
    parser.add_argument("--stage", choices=CURRICULUM_STAGES, default="target")
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--natural-curriculum", action="store_true")
    parser.add_argument(
        "--curriculum-steps",
        default=None,
        help="Comma-separated timesteps matching the selected curriculum.",
    )
    parser.add_argument("--assist-scale", type=float, default=None)
    parser.add_argument("--balance-assist-scale", type=float, default=None)
    parser.add_argument("--no-assist", action="store_true", help="Disable deterministic target assist.")
    parser.add_argument("--no-progress-bar", action="store_true")
    parser.add_argument("--dummy-vec", action="store_true", help="Use DummyVecEnv instead of SubprocVecEnv.")
    parser.add_argument("--smoke", action="store_true", help="Run a short PPO compatibility check.")
    parser.add_argument(
        "--promote-final",
        action="store_true",
        help="Also write g1_locomotion_walk_final and ppo_model_final from this run.",
    )
    return parser.parse_args()


def build_model(env, args: argparse.Namespace, ppo_kwargs: dict) -> PPO:
    return PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="cpu",
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        tensorboard_log="./tb_logs",
        policy_kwargs={"log_std_init": -1.0},
        **ppo_kwargs,
    )


def train_stage(
    *,
    args: argparse.Namespace,
    stage: str,
    timesteps: int,
    n_envs: int,
    ppo_kwargs: dict,
    warm_start_path: Path | None,
    assist_scale: float,
    balance_assist_scale: float,
) -> Path:
    if args.no_assist:
        assist_scale = 0.0
        balance_assist_scale = 0.0
    console.print(
        f"[bold cyan]Building {stage!r} G1 locomotion environments "
        f"(assist_scale={assist_scale:.2f}, balance_assist_scale={balance_assist_scale:.2f})...[/]"
    )
    for idx in range(n_envs):
        console.print(f"  {idx + 1}. {GENERATED_SCENES[idx]['name']}")

    env = build_env(
        n_envs=n_envs,
        use_subproc=not args.dummy_vec,
        stage=stage,
        assist_scale=assist_scale,
        balance_assist_scale=balance_assist_scale,
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // n_envs, 1),
        save_path=MODELS_DIR,
        name_prefix=f"{args.model_name}_{stage}",
        verbose=1,
    )

    if warm_start_path is None:
        model = build_model(env, args, ppo_kwargs)
    else:
        console.print(f"[cyan]Warm-starting {stage!r} from {warm_start_path}.zip[/]")
        model = PPO.load(warm_start_path, env=env, device="cpu")

    console.print(
        f"[bold green]Training {stage!r} PPO for {timesteps:,} steps "
        f"across {n_envs} env(s)[/]"
    )
    model.learn(
        total_timesteps=timesteps,
        callback=checkpoint_cb,
        progress_bar=not args.no_progress_bar,
        reset_num_timesteps=warm_start_path is None,
    )

    stage_path = Path(MODELS_DIR) / f"{args.model_name}_{stage}_final"
    model.save(stage_path)
    console.print(f"[bold green]Stage {stage!r} saved to {stage_path}.zip[/]")
    env.close()
    return stage_path


def main():
    args = parse_args()
    os.makedirs(MODELS_DIR, exist_ok=True)
    if args.natural_curriculum:
        args.curriculum = True
    if args.model_name is None:
        args.model_name = NATURAL_MODEL_NAME if args.natural_curriculum else DEFAULT_MODEL_NAME

    if args.smoke:
        args.timesteps = min(args.timesteps, 256)
        args.n_envs = 1
        args.model_name = f"{args.model_name}_smoke"
        args.dummy_vec = True
        args.curriculum = False
        args.natural_curriculum = False

    n_envs = max(1, min(args.n_envs, len(GENERATED_SCENES)))

    ppo_kwargs = {
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
    }
    if args.smoke:
        ppo_kwargs.update(
            {
                "n_steps": 64,
                "batch_size": 32,
                "n_epochs": 1,
            }
        )

    if args.curriculum:
        curriculum = NATURAL_CURRICULUM if args.natural_curriculum else STANDARD_CURRICULUM
        if args.curriculum_steps:
            steps = parse_curriculum_steps(args.curriculum_steps, len(curriculum))
            curriculum = tuple(
                (stage, step_count, assist_scale, balance_assist_scale)
                for (stage, _, assist_scale, balance_assist_scale), step_count in zip(curriculum, steps)
            )
        warm_start_path = None
        for stage, timesteps, assist_scale, balance_assist_scale in curriculum:
            warm_start_path = train_stage(
                args=args,
                stage=stage,
                timesteps=timesteps,
                n_envs=n_envs,
                ppo_kwargs=ppo_kwargs,
                warm_start_path=warm_start_path,
                assist_scale=assist_scale,
                balance_assist_scale=balance_assist_scale,
            )
        final_source = warm_start_path
    else:
        assist_scale = 0.0 if args.no_assist else (1.0 if args.assist_scale is None else args.assist_scale)
        balance_assist_scale = (
            0.0
            if args.no_assist
            else (assist_scale if args.balance_assist_scale is None else args.balance_assist_scale)
        )
        final_source = train_stage(
            args=args,
            stage=args.stage,
            timesteps=args.timesteps,
            n_envs=n_envs,
            ppo_kwargs=ppo_kwargs,
            warm_start_path=None,
            assist_scale=assist_scale,
            balance_assist_scale=balance_assist_scale,
        )

    final_path = Path(MODELS_DIR) / f"{args.model_name}_final"
    PPO.load(final_source, device="cpu").save(final_path)
    console.print(f"[bold green]Training complete. Model saved to {final_path}.zip[/]")

    wrote_compatibility = False
    if not args.smoke and (args.model_name == DEFAULT_MODEL_NAME or args.promote_final):
        compatibility_path = Path(MODELS_DIR) / "ppo_model_final"
        PPO.load(final_source, device="cpu").save(compatibility_path)
        console.print(f"[green]Also wrote compatibility model to {compatibility_path}.zip[/]")
        wrote_compatibility = True
    if not args.smoke and args.promote_final and args.model_name != DEFAULT_MODEL_NAME:
        promoted_path = Path(MODELS_DIR) / f"{DEFAULT_MODEL_NAME}_final"
        PPO.load(final_source, device="cpu").save(promoted_path)
        console.print(f"[green]Also promoted model to {promoted_path}.zip[/]")
        wrote_compatibility = True
    if not args.smoke and not wrote_compatibility:
        console.print("[yellow]Skipped compatibility model write for non-default model name.[/]")


if __name__ == "__main__":
    main()
