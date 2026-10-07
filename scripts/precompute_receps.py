"""
Pre-compute receptacle maps (receps.json) for all ALFWorld scenes.
This avoids the expensive explore_scene() call during training resets.

With load_receps=True and pre-computed files, OracleAgent.__init__()
loads from disk (~0.01s) instead of teleporting to every openable point
and running instance segmentation (~25-400s).

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/precompute_receps.py
"""
import os
import sys
import json
import time
from collections import defaultdict

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf
from alfworld.agents.environment import get_environment
from alfworld.agents.controller import OracleAgent


def main():
    config = OmegaConf.load("alfworld_config.yaml")
    config = OmegaConf.to_container(config, resolve=True)

    config['env']['type'] = 'AlfredThorEnv'
    env = get_environment('AlfredThorEnv')(config, train_eval='train')
    env = env.init_env(batch_size=1)

    worker = env.envs[0]

    # Group tasks by scene
    scene_tasks = defaultdict(list)
    for task_file in env.json_file_list:
        parts = task_file.split('/')
        for part in parts:
            if '-' in part:
                segments = part.split('-')
                try:
                    scene_num = int(segments[-1])
                    scene_name = 'FloorPlan%d' % scene_num
                    scene_tasks[scene_name].append(task_file)
                    break
                except (ValueError, IndexError):
                    continue

    print(f"Found {len(scene_tasks)} unique scenes across {len(env.json_file_list)} tasks")

    # Check which scenes already have receps.jsonc
    already_done = 0
    to_compute = []
    for scene_name, tasks in scene_tasks.items():
        # Pick one representative task per scene
        task_file = tasks[0]
        traj_root = os.path.dirname(task_file)
        recep_file = os.path.join(traj_root, 'receps.json')
        if os.path.isfile(recep_file):
            already_done += 1
        else:
            to_compute.append((scene_name, task_file))

    print(f"Already computed: {already_done}, need to compute: {len(to_compute)}")

    for i, (scene_name, task_file) in enumerate(to_compute):
        traj_root = os.path.dirname(task_file)
        recep_file = os.path.join(traj_root, 'receps.json')

        print(f"\n[{i+1}/{len(to_compute)}] {scene_name} ...")

        t0 = time.time()

        # Load the task
        worker.set_task(task_file)

        scene_num = worker.traj_data['scene']['scene_num']
        object_poses = worker.traj_data['scene']['object_poses']
        dirty_and_empty = worker.traj_data['scene']['dirty_and_empty']
        object_toggles = worker.traj_data['scene']['object_toggles']

        # Reset THOR to this scene
        worker.env.reset(scene_name)
        worker.env.restore_scene(object_poses, object_toggles, dirty_and_empty)
        worker.env.step(dict(worker.traj_data['scene']['init_action']))

        import alfworld.agents
        class args: pass
        args.reward_config = os.path.join(alfworld.agents.__path__[0], 'config/rewards.json')
        worker.env.set_task(worker.traj_data, args, reward_type='dense')

        # Create OracleAgent — this calls explore_scene() which is expensive
        controller = OracleAgent(
            worker.env, worker.traj_data, traj_root,
            load_receps=False, debug=False,
            goal_desc_human_anns_prob=0.0,
        )

        # Save the receptacle map
        controller.save_receps()
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s — saved {recep_file}")
        print(f"  Receptacles found: {len(controller.receptacles)}")

        # Now copy the recep file to ALL task dirs for this scene
        # (different trials of the same scene share the same FloorPlan layout)
        copied = 0
        for other_task in scene_tasks[scene_name]:
            other_root = os.path.dirname(other_task)
            other_recep = os.path.join(other_root, 'receps.json')
            if not os.path.isfile(other_recep) and other_root != traj_root:
                import shutil
                shutil.copy2(recep_file, other_recep)
                copied += 1
        print(f"  Copied receps.json to {copied} other trial dirs for {scene_name}")

    print(f"\nDone! Pre-computed receptacle maps for {len(to_compute)} scenes.")

    env.envs[0].env.stop()


if __name__ == "__main__":
    main()
