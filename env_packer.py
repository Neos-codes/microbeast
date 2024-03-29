import torch
import numpy as np
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv

import csv


def _format_obs(obs: np.ndarray) -> torch.Tensor:
    obs = torch.from_numpy(obs).float()    # int32 -> float32
    return obs.view((1, 1) + obs.shape)    # (...) -> (T, B, ...)

def _format_action_masks(action_mask: np.ndarray) -> torch.Tensor:
    action_mask = torch.from_numpy(action_mask)
    return action_mask.view((1,) + action_mask.shape)


class Env_Packer:
    """ This wrapper class packages the outputs of a gym-microRTS environment """

    def __init__(self, envs: MicroRTSGridModeVecEnv, a_id: int, exp_name: str):
        self.envs = envs
        self.n_envs = self.envs.num_envs
        self.ep_return = None    # Episode Return
        self.ep_step = None      # Episode Step

        self.a_id = a_id         # Actor id (Process id)
        self.exp_name = exp_name # Nombre experimento y csv

    def initial(self) -> dict:
        """ Initiliaze and packages the first frame of the envs game """

        env_h = self.envs.height    # Height of grid
        init_reward = torch.zeros(1, self.n_envs)
        init_last_action = torch.zeros(1, self.n_envs, 7*(env_h**2), dtype=torch.int64)
        self.ep_return = torch.zeros(1, self.n_envs, dtype=torch.uint8)

        self.ep_step = torch.zeros(1, self.n_envs, dtype=torch.uint8)
        init_dones = torch.zeros(1, self.n_envs, dtype=torch.uint8)
        init_obs = _format_obs(self.envs.reset())
        init_action_mask = _format_action_masks(self.envs.get_action_mask().reshape(self.n_envs, -1))

        ret = dict(
                obs=init_obs,
                reward=init_reward,
                done=init_dones,
                ep_return=self.ep_return,
                ep_step=self.ep_step,
                last_action=init_last_action,
                action_mask=init_action_mask
                )

        return ret


    def step(self, action: torch.Tensor) -> dict:
        """ Step into gym-microRTS environment and packages the frame info for the buffer """
        obs, reward, done, unused_info = self.envs.step(action)

        self.ep_step += 1
        self.ep_return += reward   # OJO AQUI, PUEDE MODIFICARSE PARA PPO

        ep_step = self.ep_step
        ep_return = self.ep_return


        real_dones = np.where(done == True)[0]
            #print("real_dones:", real_dones)

        if any(e == True for e in done):
            with open(self.exp_name + ".csv", "a") as data:
                csv_w = csv.writer(data)
                for i in real_dones:
                    csv_w.writerow([self.ep_return.view((-1))[i].item(), self.ep_step.view((-1))[i].item(), i])
                    self.ep_step[0][i] = 0
                    self.ep_return[0][i] = 0

        obs = _format_obs(obs)
        reward = torch.Tensor(reward).view(1, self.n_envs)
        done = done.astype("uint8")


        # Actualizar con mascara de done
        #self.ep_done = np.where(done == 1, 1, self.ep_done)   # Mascara de ambientes done
        
        done = torch.Tensor(done).view(1, self.n_envs)  # Change  done -> self.ep_done
        
        action_mask = _format_action_masks(self.envs.get_action_mask().reshape(self.n_envs, -1))


        ret = dict(
                obs=obs,
                reward=reward,
                done=done,
                ep_return=ep_return,
                ep_step=ep_step,
                last_action=action,
                action_mask=action_mask,
                )

        return ret

    def render(self) -> None:
        """ Render the first env of gym-microRTS instance """
        self.envs.render()

    def reset(self) -> dict:
        """ Reset all envs in gym-microRTS instance """
        return self.initial()

    def close(self) -> None:
        self.envs.close()
