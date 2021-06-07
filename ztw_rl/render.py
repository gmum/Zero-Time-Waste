import argparse
import os
import subprocess
from pathlib import Path
from typing import List

import neptune.new as neptune
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.vec_video_recorder import \
    VecVideoRecorder

from train import make_envs
from utils import ALGOS, get_run_by_id



def display_agent(model: BaseAlgorithm, env: VecEnv, n: int):
    obs = env.reset()
    for _ in range(n):
        action, _ = model.predict(obs, deterministic=False)
        obs, _, _, _ = env.step(action)
        env.render()


def record_agent(model: BaseAlgorithm, env: VecEnv, n: int, save_dir: Path):
    # requires: xvfb, ffmpeg... ?
    xvfb_proc = subprocess.Popen(['Xvfb', ':1', '-screen', '0', '400x300x24'])
    os.environ['DISPLAY'] = ':1'
    env = VecVideoRecorder(env, str(save_dir), record_video_trigger=lambda step: step == 0, video_length=n)
    obs = env.reset()
    for _ in range(n + 1):
        action, _ = model.predict(obs, deterministic=False)
        obs, _, _, _ = env.step(action)
    # Save the video
    env.close()
    xvfb_proc.terminate()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--model_path', help='path to the saved model', type=Path)
    # TODO save and load these automatically
    parser.add_argument('-t', '--model_type', help='model (algorithm) type', type=str, required=True)
    parser.add_argument('-e', '--env_id', help='gym environment id', type=str, required=True)
    parser.add_argument('--frame_stack', help='frame stack count', type=int, default=None)
    #
    parser.add_argument('--neptune_project', help='neptune project qualified name', type=str)
    parser.add_argument('--neptune_exp_id', help='neptune experiment id to load the model from', type=str)
    parser.add_argument('-m',
                        '--mode',
                        help='whether to display on screen or save into file',
                        choices=['render', 'save', 'save_neptune'],
                        default='render',
                        type=str)
    parser.add_argument('-s', '--timesteps', help='number of timesteps to record', default=1000, type=int)
    args = parser.parse_args()

    assert args.model_path is not None or (args.neptune_project is not None and args.neptune_exp_id is not None)

    # load model
    if args.model_path is not None:
        loaded_model = ALGOS[args.model_type].load(args.model_path)
    elif args.neptune_project is not None:
        run = get_run_by_id(args.neptune_project, args.neptune_exp_id)
        # download model
        dest_path = Path('/tmp') / 'best_model.zip'
        run['best_model'].download(dest_path)
        loaded_model = ALGOS[args.model_type].load(dest_path)
        dest_path.unlink()
    env, eval_env = make_envs(args.env_id, 1, args.frame_stack)

    if args.mode == 'render':
        display_agent(loaded_model, eval_env, args.timesteps)
    elif 'save' in args.mode:
        video_save_dir = Path.cwd() / 'agent_videos'
        video_save_dir.mkdir(exist_ok=True)
        record_agent(loaded_model, eval_env, args.timesteps, video_save_dir)
        # if 'neptune' in args.mode:
        # TODO find file and upload?
        # video_save_dir.glob()?
        # run['video_sample'].upload_video()?


if __name__ == '__main__':
    main()
