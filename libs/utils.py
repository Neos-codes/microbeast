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

def vtrace(batch: dict, device="cpu"):
    
    print("baseline batch size:", batch["baseline"].size())
    print("ep steps:", batch["ep_step"])
    n_envs, unroll_length = batch["reward"].size()
#bootstrap_v = batch["baseline"]


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
        if full_queue.qsize() < 4:
            #print("full_queue size:", full_queue.qsize())
            continue
        else:
            break

    # Obtener indices de la full queue para actualizar
    with lock:
        indices = [full_queue.get() for _ in range(batch_size)]


    # Guardar las trayectorias obtenidas en un batch
    # Es una lista de stacks que contienen los steps almacenados en buffer
    batch = {
            key: torch.stack([buffers[key][m] for m in indices], dim=1) for key in buffers
    }

    # Liberar indices obtenidos de la full queue en la free queue
    # No es necesario el lock porque no hay indices iguales
    for m in indices:
        #print(f"Buffer {m} liberado!")
        free_queue.put(m)
    
    # Pasar tensores del batch al device de entrenamiento
    #batch = {k: t.to(device=device, non_blocking=True) for k, t in batch.items()}
    
    # ----- reshape batch para simplificar codigo ----- #
    # "shapes" es un dict definido en los inicios de la funcion "train()"
    # con el fin de reducir los parametros de esta funcion

    batch["reward"] = batch["reward"].permute((1, 2, 0)).reshape(shapes["batch_envs"])
    batch["obs"] = batch["obs"].permute((1, 2, 0, 3, 4, 5)).reshape(shapes["obs"])

    batch["done"] = batch["done"].permute((1, 2, 0)).reshape(shapes["batch_envs"]).float()
    batch["ep_return"] = batch["ep_return"].permute((1, 2, 0)).reshape(shapes["batch_envs"])
    batch["ep_step"] = batch["ep_step"].permute((1, 2, 0)).reshape(shapes["batch_envs"])
    batch["policy_logits"] = batch["policy_logits"].permute((1, 2, 0, 3)).reshape(shapes["logits"])
    batch["action_mask"] = batch["action_mask"].permute((1, 2, 0, 3)).reshape(shapes["action_mask"])
    batch["baseline"] = batch["baseline"].permute((1, 2, 0)).reshape(shapes["batch_envs"])
    batch["logprobs"] = batch["logprobs"].permute((1, 2, 0)).reshape(shapes["batch_envs"])
    batch["action"] = batch["action"].permute((1, 2, 0, 3)).reshape(shapes["action"])
    batch["last_action"] = batch["last_action"].permute((1, 2, 0, 3)).reshape(shapes["action"])

    """print("In batch:")
    for key in batch:
        print(f"{key}:", batch[key].size())"""

    print("Done shape before flatten:", batch["done"].size())
    print(batch["done"][0])

    # Este arreglo tendrá solo los pasos dados por cada ambiente
    batch["ep_step"] = batch["ep_step"][:, 0]
    print("kaka size:", batch["ep_step"].size()[0])
    print("ep_step sum:", batch["ep_step"].sum())
    
    
    batch["discounts"] = torch.arange(1, batch["ep_step"][0] + 1)
    for i in range(1, batch["ep_step"].size()[0]):
       batch["discounts"] = torch.cat((batch["discounts"], torch.arange(1, batch["ep_step"][i])))
    batch["discounts"] = batch["discounts"]*0.99 


    #print("Policy logits size:", batch["policy_logits"].size())


    # Obtenemos los valores de ventaja con las dimensiones actuales del batch
    advantages = get_advantages(batch, gamma)#, device)
    #deltas = get_deltas(batch, gamma)

    # Luego estiramos todas las features
    #return flatten_batch_and_advantages(batch, advantages, shapes)  #dict, tensor 
    return flatten_batch_and_advantages(batch, advantages, shapes)



# TERMINAR ESTO!
def PPO_learn(actor_model: nn.Module, learner_model, exp_name: str, batch, advantages, optimizer, B: int, n_envs: int, lock=threading.Lock()):
    
    """ Performs a learning (optimization) step """

    exp_name = exp_name + ".csv"

    ep_returns = []
    mean_loss = []
    with lock:
        # Debo calcular la recompensa del ultimo step
        #learner_outputs, _ = learner_model.get_action(batch, agent_state=())

        batch_size = batch["reward"].size()[-1]  # = n_envs*B*T
        unroll_length = batch_size // (B*n_envs)

        """print("obs shape:", batch["obs"].size())
        inds = np.arange(batch_size)
        learner_output = learner_model.get_action(batch, inds=inds, agent_state=())"""

        inds = []
        count = 0
        #print(batch["ep_step"])
        # Esto trunca el batch contando los indices de los steps hasta el final
        # de la primera partida de cada ambiente
        print("ep step:", batch["ep_step"])
        for i in range(0, batch_size, unroll_length):
            print(batch["ep_step"][count])
            inds += list(range(i, i+batch["ep_step"][count]))
            ep_returns.append(batch["ep_return"][0][i+batch["ep_step"][count]])
            count += 1

        #print("ep_returns:", ep_returns)
        
        # ahora batch_size solo cuenta los indices de trayectorias validas
        batch_size = len(inds)
        #inds = np.arange(batch_size)  # arange of (n_envs*B*T)
        

        # Por cada batch, actualizamos 4 veces
        print("Updating...")
        for e in range(4):
            #print(f"Update {e}")
            
            learner_output = learner_model.get_action(batch, inds=inds, agent_state=())
            ratio = learner_output[0]["logprobs"] - batch["logprobs"][0, inds, ...]


            

            #print("learner logprob size:", learner_output[0]["logprobs"].size())
            #print("batch logprob size:", batch["logprobs"][0, inds, ...].size())
            #print("Ratio size:", ratio.size())
            #clipped_ratio = 
            with torch.no_grad():
                ratio_exp = ratio.exp()
                cs = torch.clamp(ratio_exp, max=1.0)
                ratio_exp = torch.clamp(ratio_exp, max=1.0)

                deltas = ratio * (batch["reward"][0, inds] * batch["baseline"][0, inds] - batch["baseline"][0, inds])
                print("deltas size:", deltas.size())
                print("baseline size:", batch["baseline"][0, inds, ...].size())

                acc = torch.zeros(1)
                result = []
                count = 0
                for i in range(batch["ep_step"].size()[0]):
                    for t in range(count + batch["ep_step"][i] - 1, count-1, -1):
                        acc = deltas[t] + batch["discounts"][t] * cs[t] * acc
                        result.append(acc)
                    count += batch["ep_step"][i] - 1

                result.reverse()
                v_s_ = torch.tensor(result, dtype=torch.float32)
                v_s = v_s_ + batch["baseline"][0, inds, ...]

            pg_loss = ratio_exp * batch["logprobs"][0, inds, ...] * (batch["reward"][0, inds, ...] + batch["discounts"] * v_s - batch["baseline"][0, inds, ...])

            value_loss = v_s - batch["baseline"][0, inds, ...]

            entropy_loss = torch.sum(batch["logprobs"][0, inds, ...] * batch["entropy"][0, inds, ...])



                print("ratio new shape:", ratio.size())


            # Total loss
            loss = ratio.mean()*deltas.mean() 
            print("loss:", loss)
            #print("rewards shape:", batch["reward"].size())    #[0, inds]
            #print("baseline shape:", batch["baseline"].size()) #[0, inds]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            actor_model.load_state_dict(learner_model.state_dict())

            #mean_loss.append(loss.item())
    
    print("Mean loss:", loss)
    #print("Mean episode return:", np.array(ep_returns).mean())
    print("Ep steps:", np.array(batch["ep_step"]).mean())
