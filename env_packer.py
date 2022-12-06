import torch
import numpy as np
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv


def _format_obs(obs: np.ndarray) -> torch.Tensor:
    obs = torch.from_numpy(obs).float()    # int32 -> float32
    return obs.view((1, 1) + obs.shape)    # (...) -> (T, B, ...)

def _format_action_masks(action_mask: np.ndarray) -> torch.Tensor:
    action_mask = torch.from_numpy(action_mask)
    return action_mask.view((1,) + action_mask.shape)


class Env_Packer:
    """ This wrapper class packages the outputs of a gym-microRTS environment """

    def __init__(self, envs: MicroRTSGridModeVecEnv, a_id: int):
        self.envs = envs
        self.n_envs = self.envs.num_envs
        self.ep_return = None    # Episode Return
        self.ep_step = None      # Episode Step
        self.ep_done = None      # Episonde Done

        self.a_id = a_id         # Actor id (Process id)

    def initial(self) -> dict:
        """ Initiliaze and packages the first frame of the envs game """

        env_h = self.envs.height    # Height of grid
        init_reward = torch.zeros(1, self.n_envs)
        init_last_action = torch.zeros(1, self.n_envs, 7*(env_h**2), dtype=torch.int64)
        self.ep_return = torch.zeros(1, self.n_envs)
        self.ep_step = torch.zeros(1, self.n_envs, dtype=torch.int32)
        self.ep_done = np.zeros((1, self.n_envs), dtype="uint8")
        init_dones = torch.ones(1, self.n_envs, dtype=torch.uint8)
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
        # ---------
        ep_step = self.ep_step + 1
        ep_return = self.ep_return + reward
        #self.ep_return += reward   # OJO AQUI, PUEDE MODIFICARSE PARA PPO
        #ep_return = self.ep_return
        # --------

        obs = _format_obs(obs)
        reward = torch.Tensor(reward).view(1, self.n_envs)
        done = done.astype("uint8")

        # Ver si algun ambiente termina
        if any(d == 1 for d in done) and self.a_id == 0:
            print(f"{self.a_id} done:", done, "en step:", self.ep_step)

        # Actualizar con mascara de done
        self.ep_done = np.where(done == 1, 1, self.ep_done)   # Mascara de ambientes done
        self.ep_step = torch.where(torch.from_numpy(self.ep_done) == 0, ep_step, self.ep_step)
        self.ep_return = torch.where(torch.from_numpy(self.ep_done) == 0, ep_return, self.ep_return)


        #if self.a_id == 0:
            #print("done:", done)
            #print("ep_done:", self.ep_done)
        # Ver si todos los ambientes terminaron al menos una vez
        if np.all(self.ep_done == 1) and self.a_id == 0:
            print("Todos los ambientes done!")


        done = torch.Tensor(self.ep_done).view(1, self.n_envs)  # Change  done -> self.ep_done

        
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
