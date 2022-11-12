import numpy as np
import multiprocessing as mp
import torch
from parser import *
import typing

from libs.utils import create_buffers, create_env
from model import Agent
from env_packer import Env_Packer

# TO DO: Funcion Act() donde se crea un Agente nuevo y una NN nueva para entrenar
#   - Quede en empaquetar un ambiente, falta verificar que funcione el
#     metodo step


# NOTA: Si bien los buffers y modelos no tienen el mismo id, comparten
#       los tensores entre procesos



# Un Buffer será un diccionario con keys string y claves list(Tensor)
Buffers = typing.Dict[str, typing.List[torch.Tensor]]

def act(model: Agent, buffers: Buffers, num: int):
    print(f"Hola! Soy el actor {num}")
    gym_envs = create_env(8, 2, 512)
    envs = Env_Packer(gym_envs)

def train():
    print("Training...")

    # ----- temp ----- #
    n_actors = 4       # num of actors (subprocesses) training
    n_envs = 2         # num envs per gym instance
    env_size = 8       # options: [8, 10]  grid size: (8x8), (10x10)
    T = 80             # unroll_length
    B = 4              # batch_size 
    n_buffers = max(2 * n_actors, B) # Como minimo, el doble de actores

    # Crear ambiente para obtener shapes
    micro_env = create_env(env_size, n_envs, 512) 
    obs_space_shape = micro_env.observation_space.shape
    nvec = micro_env.action_space.nvec.tolist()
    micro_model = Agent(micro_env.observation_space.shape, nvec, 8*8)
    micro_model.share_memory()
    print(f"Model id in main process: {id(micro_model)}")

    # Crear buffer
    buffers = create_buffers(n_buffers, n_envs, T, micro_env.observation_space.shape)
    print(f"Buffers id in main process: {id(buffers)}")

    # Desde ahora, micro_env solo sirve para constructor del Learner

    # ----- Crear Queues ----- #
    # free_queue para indices de buffers disponibles para llenar
    # full_queue para indices de buffers con trayectorias para backprop
    ctx = mp.get_context("spawn")
    free_queue = ctx.SimpleQueue()
    full_queue = ctx.SimpleQueue()

    # ----- Crear subprocesos (Actores) ----- #
    actor_processes = []
    for i in range(n_actors):
        actor = ctx.Process(
                    target=act,
                    args=(micro_model, buffers, i)
                )
        actor.start()
        actor_processes.append(actor)

    
    # ----- Crear Modelo de Aprendizaje ----- #
    micro_learner = Agent(obs_space_shape, nvec, 8*8)

    # ----- Crear Optimizador ----- #
    micro_optimizer = torch.optim.Adam(micro_learner.parameters(), lr=2.5e-4, eps=1e-5)

    # ----- Llenar la free queue con los indices ----- #
    for index in range(n_buffers):
        free_queue.put(index)   # Indices de las trayectoras para backprop


def test():
    print("Testing...")


def main():
    if args.test:
        test()
    else:
        train()

if __name__ == "__main__":
    main()
