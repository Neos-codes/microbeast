import os
from time import sleep
import numpy as np
import multiprocessing as mp
import threading
import torch
from parser import *
import typing

from libs.utils import create_buffers, create_env, get_batch, get_advantages, PPO_learn 
from model import Agent 
from env_packer import Env_Packer


os.environ["OMP_NUM_THREADS"] = "1"  # Para multitrheading



# NOTA: Si bien los buffers y modelos no tienen el mismo id, comparten
#       los tensores entre procesos



# Un Buffer será un diccionario con keys string y claves list(Tensor)
Buffers = typing.Dict[str, typing.List[torch.Tensor]]




def act(agent: Agent,                   # nn.Module
        buffers: Buffers,               # Buffer dictionary
        a_id: int,                      # Agent id number
        free_queue: mp.SimpleQueue,     # Queue of free indexes from buffer
        full_queue: mp.SimpleQueue,     # Que of full indexes from buffer
        unroll_length: int,             # T, unroll_size, tamaño trayectoria  (see IMPALA paper)
        ):

    print(f"Hola! Soy el actor {a_id}")
    gym_envs = create_env(8, 2, 512)
    envs = Env_Packer(gym_envs)
    env_output = envs.initial()
    agent_output, _ = agent.get_action(env_output)
     

    # Actuar indefinidamente en el ambiente
    while True:
        # Mostrar tablero
        envs.render()
        # Obtener index de la free queue
        index = free_queue.get() 
        print(f"index in actor {a_id}: {index}")

        # Guardar a_id de actor para referenciar el modelo al
        # hacer backpropagate
        print("Actor id:", a_id)
        buffers["actor"][index] = a_id


        # ----- Guardar condiciones iniciales en el buffer ----- #
        # Del ambiente
        for key in env_output:
            # Posicion 0 de la trayectoria en el buffer
            buffers[key][index][0, ...] = env_output[key]
        # Del agente
        for key in agent_output:
            buffers[key][index][0, ...] = agent_output[key]
        
        # Sleep "entre buffers"
        sleep(5)
        if free_queue.empty():
            print(f"Continue por index ocupados en actor {a_id}")
            continue

        for t in range(unroll_length):
            envs.render()
            # Generar accion del agente
            with torch.no_grad():
                agent_output, _ = agent.get_action(env_output)

            
            # Step en el ambiente
            env_output = envs.step(agent_output["action"])


            # Guardar step en el buffer 
            for key in env_output:
                buffers[key][index][t + 1, ...] = env_output[key]
               
            # Del agente
            for key in agent_output:
                buffers[key][index][t + 1 , ...] = agent_output[key]

            

        # Una vez llena la trayectoria, pasar a full queue
        full_queue.put(index)
        # Para experimentar vuelvo a pasarlo a la free_queue
                 


def train():
    print("Training...")

    # ----- temp ----- #
    n_actors = 4       # num of actors (subprocesses) training
    n_envs = 2         # num envs per gym instance
    env_size = 8       # options: [8, 10]  grid size: (8x8), (10x10)
    T = 80             # unroll_length
    B = 4              # batch_size 
    gamma = 0.99       # Discount factor
    n_learner_threads = 2   # Cuantos threads learners tendremos
    n_buffers = max(2 * n_actors, B) # Como minimo, el doble de actores
    train_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Crear ambiente para obtener shapes
    micro_env = create_env(env_size, n_envs, 512) 
    obs_space_shape = micro_env.observation_space.shape
    nvec = micro_env.action_space.nvec.tolist()
    micro_model = Agent(micro_env.observation_space.shape, nvec, 8*8)
    micro_model.share_memory()

   
    # Diccionario con shapes para simplificar paso de parametros
    shapes = dict(obs=(n_envs*B, T+1,) + micro_env.observation_space.shape,
                  action=(n_envs*B, T+1,) + micro_env.action_space.shape,
                  logits=(n_envs*B, T+1,) + (sum(micro_model.nvec),),
                  action_mask=(n_envs*B, T+1,) + (sum(micro_model.nvec),),
                  batch_envs=(n_envs*B, -1)
                  )

    #print("shapes:\n", shapes, shapes["obs"])

    # Crear buffer
    buffers = create_buffers(n_buffers, n_envs, B, T, micro_env.observation_space.shape)

    # Desde ahora, micro_env solo sirve para constructor del Learner
    print("actor ids:", buffers["actor"])

    print("Main process pid:", os.getpid())

    # ----- Crear Queues ----- #
    # free_queue para indices de buffers disponibles para llenar
    # full_queue para indices de buffers con trayectorias para backpropagation
    ctx = mp.get_context("spawn")
    free_queue = ctx.SimpleQueue()
    full_queue = ctx.SimpleQueue()
    
    # ----- Llenar la free queue con los indices ----- #
    for index in range(n_buffers):
        free_queue.put(index)   # Indices de las trayectoras para backprop


    # ----- Crear subprocesos (Actores) ----- #
    actor_processes = []
    actor_models = []
    for i in range(n_actors):

        # Creamos modelo para pasarlo al proceso
        model = Agent(micro_env.observation_space.shape, nvec, env_size**2)
        
        # Creamos proceso actor con el modelo recien creado
        actor = ctx.Process(
                    target=act,
                    args=(model, buffers, i, free_queue, full_queue, T)
                )
        actor.start()

        # Guardamos referencias al proceso actor y a su modelo de red neuronal
        actor_processes.append(actor)
        actor_models.append(model)

    
    # ----- Crear Modelo de Aprendizaje ----- #
    micro_learner = Agent(obs_space_shape, nvec, env_size**2)

    # ----- Crear Optimizador ----- #
    micro_optimizer = torch.optim.Adam(micro_learner.parameters(), lr=2.5e-4, eps=1e-5)

    

    # Step se comppartira entre threads, para contar cuantos steps llevamos
    # hasta alcanzar el maximo y terminar de entrenar
    step = 0
    # Definir funcion batch_and_learn()
    # Es dentro de la funcion para compartir variables globales

    def batch_and_learn(i: int, total_steps: int, lock=threading.Lock()):
        """ Get batches from buffer and backpropagates the info into NN model """

        #nonlocal step  # Para referenciar la variable step de train()
        #nonlocal gamma # Para referenciar el factor de descuento de train()
        #nonlocal micro_env # Para referenciar un ambiente y sacar features
        #print("Gamma nonlocal:", gamma)

        #while step < total_steps:
            

            # Generar batches
            #batch = get_batch(B, gamma, train_device, free_queue, full_queue, buffers)
           
            #PPO_learn(

        print("PID en batch_and_learn", os.getpid())


        # END BATCH AND LEARN

    # Proceso padre no puede terminar antes que los hijos o da error
    """print("Antes de dormir...")
    print("Obs space shape:", micro_env.observation_space.shape)
    print("Action space shape:", micro_env.action_space.shape)"""
    sleep(30)
    print("Buffers ocupados por:", buffers["actor"])
    #print("Getting batches!")
    """print("Antes del batch:")
    for key in buffers:
        print(key, "shape:", buffers[key][0].size())
    batch, advantages = get_batch(B, shapes, gamma, train_device, free_queue, full_queue, buffers)
    
    for key in batch:
        print(key, "shape:", batch[key].size())

    print("Advantages shape:", advantages.size())
    print("batch_size:", batch["reward"].size()[-1])
    #print("Unroll size:", batch["reward"].size())"""

    batch_and_learn(2, 2)


def test():
    print("Testing...")


def main():
    if args.test:
        test()
    else:
        train()

if __name__ == "__main__":
    main()
