import torch
import torch.nn as nn
import typing
import numpy as np
import multiprocessing as mp
import threading

from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv

import random
# ----- Aliases
# Un Buffer será un diccionario con keys string y claves list(Tensor)
Buffers = typing.Dict[str, typing.List[torch.Tensor]]


# ----- Functions
# - create_buffers(): Creates {n_buffer} buffers that share memory between processes

# - create_env(): Returns a gym-MicroRTS env of (size x size) and \"n_envs\" bots simultaneously

# - get_batch(): Returns {batch_size} batches taken from the buffer 

# - learn(): Take unroll from batch and makes backpropagation in model

# ----- Functions ----- #

# T = unroll_length    n_envs: num of envs in a gym-microRTS execution 
def create_buffers(n_buffers: int, n_envs: int, B: int, T: int, obs_size: tuple) -> Buffers:
    """Creates \"n_buffers\" buffers that share memory between processes"""

    print("Creando buffers...")
    h = w = obs_size[0]
    specs = dict(
        obs=dict(size=(T + 1, n_envs, *obs_size), dtype=torch.float32),
        reward=dict(size=(T + 1, n_envs), dtype=torch.float32),
        done=dict(size=(T + 1, n_envs), dtype=torch.bool),
        ep_return=dict(size=(T + 1, n_envs), dtype=torch.float32),
        ep_step=dict(size=(T + 1, n_envs), dtype=torch.int32),
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



def create_env(size: int, n_envs: int, max_steps: int) -> MicroRTSGridModeVecEnv:
    """ Returns a gym-MicroRTS env of (size x size) and \"n_envs\" bots simultaneously"""
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

def flatten_batch_and_advantages(batch: dict, advantages: torch.Tensor, shapes: dict) -> (dict, torch.Tensor):

    """ Takes the batch and advantages and flatten them """


    # Batch flatten 
    batch["reward"] = batch["reward"].reshape((1, -1))
    batch["obs"] = batch["obs"].reshape((1, -1,) + shapes["obs"][2:])

    batch["done"] = batch["done"].reshape((1, -1)).float()
    batch["ep_return"] = batch["ep_return"].reshape((1, -1))
    batch["ep_step"] = batch["ep_step"].reshape((-1,))
    batch["policy_logits"] = batch["policy_logits"].reshape((1, -1,) + shapes["logits"][2:])
    batch["action_mask"] = batch["action_mask"].reshape((1, -1,) + shapes["action_mask"][2:])
    batch["baseline"] = batch["baseline"].reshape((1, -1))
    batch["logprobs"] = batch["logprobs"].reshape((1, -1))
    batch["action"] = batch["action"].reshape((1, -1,) + shapes["action"][2:])
    batch["last_action"] = batch["last_action"].reshape((1, -1,) + shapes["action"][2:])
    
    
    # Advantages flatten
    advantages = advantages.flatten()

    return batch, advantages


def get_deltas(batch: dict, gamma: float, device="cpu") -> torch.Tensor:

    """ Get a batch and return a tensor with IMPALA discounted deltas """

    n_envs, unroll_length = batch["reward"].size()
    # unroll_length = T+1 in this case, cos reward shape is T+1

    deltas = torch.zeros((n_envs, unroll_length))
    
    # Aliases
    reward = batch["reward"]
    value = batch["baseline"]
    done = batch["done"].type(torch.float32)

    # ones to get result of multidmensional (1 - done)
    ones = torch.ones(n_envs, 1)

    with torch.no_grad():
        
       for t in range(unroll_length - 1):
            deltas[:, t:t+1] = reward[:, t:t+1] + gamma*value[:, t+1:t+2] * (ones - done[:, t:t+1]) - value[:, t:t+1]

    print("GET_DELTAS: deltas size ->", deltas.size()) 
    return deltas


def get_advantages(batch: dict, gamma: float, device="cpu") -> torch.Tensor:

    """ Get a batch and return a tensor with advantages for all envs with shape (n_envs, unroll_length) """

    n_envs, unroll_length = batch["reward"].size()
    # unroll_lenght = T+1 in this case, cause reward shape is T+1

    # Prepare advantages tensor
    advantages = torch.zeros((n_envs, unroll_length))

    # Aliases
    reward = batch["reward"]
    value = batch["baseline"]
    done = batch["done"].type(torch.float32)

    # Ones to get result of multidimensional 1 - done
    ones = torch.ones((n_envs, 1))#.to(device)


    # Actual calculo O(n²), puede mejorarse a O(n)
    with torch.no_grad():
        # Para cada paso de las trayectorias
        for t in range(unroll_length - 1):
            discount = 1
            advantage_t = 0
            
            # Se calcula la ventaja con todos los pasos que le suceden
            for k in range(t, unroll_length - 1):
                advantage_t += discount*(reward[:, k:k+1] + gamma*value[:, k+1:k+2] * (ones - done[:, k:k+1]) - value[:, k:k+1])
                discount *= gamma

            advantages[:, t:t+1] = advantage_t
        
    return advantages.to(device)


def get_batch(
        batch_size: int,
        shapes: dict,
        gamma: float,
        device: str,
        free_queue: mp.SimpleQueue,
        full_queue: mp.SimpleQueue,
        buffers: Buffers,
        lock=threading.Lock(),
        ) -> (dict, torch.Tensor):

    """ Returns {batch_size} batches taken from the buffer and a tensor with advantages """

    #print("Gettin batch...\nBatch size:", batch_size)


    # Iterar mientras se llena la full queue
    while True:
        if full_queue.qsize() < batch_size:
            #print("full_queue size:", full_queue.qsize())
            continue
        else:
            break

    # Obtener indices de la full queue para actualizar
    with lock:
        indices = [full_queue.get() for _ in range(batch_size)]


    # Guardar las trayectorias obtenidas en un batch
    # Concatenar trayectorias
    batch = {
            key: torch.stack([buffers[key][m] for m in indices], dim=1) for key in buffers
    }

    # dims [T, n_envs, ...] -> [T*n_envs, ...]
    for key in batch:
        last_dims = tuple(batch[key].size()[3:])
        batch[key] = batch[key].reshape((-1,)+last_dims)
    

    # Liberar indices obtenidos de la full queue en la free queue
    # No es necesario el lock porque no hay indices iguales
    for m in indices:
        #print(f"Buffer {m} liberado!")
        free_queue.put(m)
    
    # Pasar tensores del batch al device de entrenamiento
    #batch = {k: t.to(device=device, non_blocking=True) for k, t in batch.items()}
    
    return batch



# TERMINAR ESTO!
def PPO_learn(actor_model: nn.Module, learner_model, exp_name: str, batch, optimizer, B: int, n_envs: int, lock=threading.Lock()):
    
    """ Performs a learning (optimization) step """

    exp_name = exp_name + "Losses.csv"

    ep_returns = []
    mean_loss = []
    with lock:

        # Usar los ep steps para meter las observaciones al learner
        learner_outputs, _ = learner_model.get_action(batch, learning=True, inds=[], agent_state=())

        # dims [1, ...] -> [...]
        learner_outputs["baseline"] = learner_outputs["baseline"].flatten()

        bootstrap_value = learner_outputs["baseline"][-1:]

        # Move obs[t] -> action[t] to action[t] -> obs[t]
        # Quitar primer elemento del batch
        batch = {key: tensor[1:] for key, tensor in batch.items()}
        # Quitar ultimo elemento del batch
        learner_outputs = {key: tensor[:-1] for key, tensor in learner_outputs.items()}


        discounts = (~batch["done"]).float() * 0.99  # gamma = 0.99

        # VTrace
        old_logprobs = batch["logprobs"]
        new_logprobs = learner_outputs["logprobs"]
        values = learner_outputs["baseline"]
        rewards = batch["reward"] 

        ratio = new_logprobs - old_logprobs

        # --- Importance Weights
        with torch.no_grad():
            ratio_exp = ratio.exp()
            
            # rho -> Greek letter "p"    rhos in plural
            rhos = torch.clamp(ratio_exp, max=1.0)
            cs = torch.clamp(ratio_exp, max=1.0)
            
            values_t_plus_one = torch.cat([values[1:], bootstrap_value])

            deltas = rhos * (rewards + discounts * values_t_plus_one - values)

            acc = torch.zeros_like(bootstrap_value)
            result = []
            for t in range(discounts.shape[0] - 1, -1, -1):
                acc = deltas[t] + discounts[t] * cs[t] * acc
                result.append(acc)

            result.reverse()
            vs_minus_v_xs = torch.tensor(result)

            # Sumar V(xs) para tener v_s
            vs = vs_minus_v_xs + values
            
            # Advantage for policy gradients
            broadcasted_bootstrap_values = torch.ones_like(vs[0]) * bootstrap_value
            vs_t_plus_1 = torch.cat([vs[1:], broadcasted_bootstrap_values])

            clipped_pg_rhos = torch.clamp(rhos, max=1.0)

            pg_advantages = clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values)

        pg_loss = (learner_outputs["logprobs"] * pg_advantages).mean()
        print("----------\npg loss:", pg_loss)

        value_loss = 0.5*((vs - learner_outputs["baseline"])**2).mean()
        print("value loss:", value_loss)

        entropy_loss = learner_outputs["entropy"].mean()
        print("Entropy loss:", entropy_loss, "\n----------")

        total_loss = pg_loss + value_loss - 0.01*entropy_loss

        
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        actor_model.load_state_dict(learner_model.state_dict())


        print("loss:", total_loss)

        return [pg_loss.item(), value_loss.item(), entropy_loss.item(), total_loss.item()]
