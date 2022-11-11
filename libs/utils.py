import torch
import typing
import numpy as np

from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv

Buffers = typing.Dict[str, typing.List[torch.Tensor]]

# T = unroll_length    n_envs: num of envs in a gym-microRTS execution 
def create_buffers(n_buffers: int, n_envs: int, T: int, obs_size: tuple):
    print("Creando buffers...")
    h = w = obs_size[0]
    specs = dict(
        frame=dict(size=(T + 1, n_envs, *obs_size), dtype=torch.float32), # frame es equivalente a obs
        reward=dict(size=(T + 1, n_envs), dtype=torch.float32),
        done=dict(size=(T + 1, n_envs), dtype=torch.bool),
        episode_return=dict(size=(T + 1, n_envs), dtype=torch.float32),
        episode_step=dict(size=(T + 1,), dtype=torch.int32),
        policy_logits=dict(size=(T + 1, n_envs, h*w*78), dtype=torch.float32),
        baseline=dict(size=(T + 1, n_envs), dtype=torch.float32),          # baseline es V(s)
        last_action=dict(size=(T + 1, n_envs, h*w*7), dtype=torch.int64),
        action=dict(size=(T + 1, n_envs, h*w*7), dtype=torch.int64),
        action_mask=dict(size=(T + 1, n_envs, 78*h*w), dtype=torch.uint8),
        logprobs=dict(size=(T + 1, n_envs), dtype=torch.float32),
    )
    #print("specs keys:", list(specs.values()))
    #print("specs:", specs)
    buffers: Buffers = {key: [] for key in specs}
    for _ in range(n_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())
    return buffers

def create_env(size: int, n_envs: int, max_steps: int):
    print("Creando ambiente gym-MicroRTS...")
    envs = MicroRTSGridModeVecEnv(
            num_selfplay_envs=0,
            num_bot_envs=n_envs,
            max_steps=max_steps,
            render_theme=2,
            ai2s=[microrts_ai.coacAI for _ in range(n_envs)],
            map_paths=[f'maps/{size}x{size}/basesWorkers{size}x{size}.xml'],
            reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
            )
    return envs

