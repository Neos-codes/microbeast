import os
from time import sleep, perf_counter
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
    envs = Env_Packer(gym_envs, a_id)

    # Obtener info del step 0
    env_output = envs.initial()
    agent_output, _ = agent.get_action(env_output)
     
    envs_done = False
    # Actuar indefinidamente en el ambiente
    while True:
        # Mostrar tablero
        envs.render()
        # Obtener index de la free queue
        if free_queue.empty():
            #print("free queue size:", free_queue.qsize())
            #print("full queue size:", full_queue.qsize())
            continue


        index = free_queue.get() 

        if index is None:
            break
        #print(f"index in actor {a_id}: {index}")


        # ----- Guardar condiciones iniciales en el buffer ----- #
        # Del ambiente
        for key in env_output:
            # Posicion 0 de la trayectoria en el buffer
            buffers[key][index][0, ...] = env_output[key]
        # Del agente
        for key in agent_output:
            buffers[key][index][0, ...] = agent_output[key]
        

        # Ahora obtener trayectoria desde t = 1 hasta t = T+1
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

            # Si todos los ambientes terminan antes de que los pasos maximos se completen
            if torch.all(buffers["done"][index][t+1, ...] == True):
                # Se guardan en que step terminaron
                buffers["ep_step"][index][0, ...] = buffers["ep_step"][index][t+1, ...]
                envs_done = True
                break

        
        # Si entramos aqui, es porque algun ambiente no terminó
        if not envs_done:
            buffers["ep_step"][index][0, ...] = buffers["ep_step"][index][unroll_length, ...]

        # Reiniciamos el ambiente
        env_output = envs.reset()
        agent_output, _ = agent.get_action(env_output)

        # Una vez llena la trayectoria, pasar a full queue
        full_queue.put(index)
        # Para experimentar vuelvo a pasarlo a la free_queue
                 


def train(exp_name: str):
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
    max_steps = 10000000
    train_device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Train device:", train_device)

    # Crear ambiente para obtener shapes
    micro_env = create_env(env_size, n_envs, 512) 
    obs_space_shape = micro_env.observation_space.shape
    nvec = micro_env.action_space.nvec.tolist()
    micro_model = Agent(micro_env.observation_space.shape, nvec, env_size**2)
    micro_model.share_memory()

    print("Micro model device:", next(micro_model.parameters()).device)
   
    # Diccionario con shapes para simplificar paso de parametros
    shapes = dict(obs=(n_envs*B, T+1,) + micro_env.observation_space.shape,
                  action=(n_envs*B, T+1,) + micro_env.action_space.shape,
                  logits=(n_envs*B, T+1,) + (sum(micro_model.nvec),),
                  action_mask=(n_envs*B, T+1,) + (sum(micro_model.nvec),),
                  batch_envs=(n_envs*B, -1))

    #print("shapes:\n", shapes, shapes["obs"])

    # Crear buffer
    buffers = create_buffers(n_buffers, n_envs, B, T, micro_env.observation_space.shape)


    print("Main process pid:", os.getpid())

    # ----- Crear Queues ----- #
    # free_queue para indices de buffers disponibles para llenar
    # full_queue para indices de buffers con trayectorias para backpropagation
    ctx = mp.get_context("spawn")
    free_queue = ctx.Queue()
    full_queue = ctx.Queue()
    
    # ----- Llenar la free queue con los indices ----- #
    for index in range(n_buffers):
        free_queue.put(index)   # Indices de las trayectoras para backprop


    # ----- Crear subprocesos (Actores) ----- #
    actor_processes = []
    for i in range(n_actors):

        
        # Creamos proceso actor con el modelo recien creado
        actor = ctx.Process(
                    target=act,
                    args=(micro_model, buffers, i, free_queue, full_queue, T)
                )
        actor.start()

        # Guardamos referencias al proceso actor y a su modelo de red neuronal
        actor_processes.append(actor)

    
    # ----- Crear Modelo de Aprendizaje ----- #
    micro_learner = Agent(obs_space_shape, nvec, env_size**2, train_device)
    #micro_learner.cuda()
    #micro_learner.my_device()

    # ----- Crear Optimizador ----- #
    micro_optimizer = torch.optim.Adam(micro_learner.parameters(), lr=2.5e-4, eps=1e-5)

    

    # Step se comppartira entre threads, para contar cuantos steps llevamos
    # hasta alcanzar el maximo y terminar de entrenar
    step = 0
    # Definir funcion batch_and_learn()
    # Es dentro de la funcion para compartir variables globales

    def batch_and_learn(i: int, total_steps: int, exp_name: str, lock=threading.Lock()):
        """ Get batches from buffer and backpropagates the info into NN model """

        nonlocal step  # Para referenciar la variable step de train()
        nonlocal gamma # Para referenciar el factor de descuento de train()
        nonlocal micro_env # Para referenciar un ambiente y sacar features

        while step < total_steps:
            
            print(f"Current updating step: {step}")
            start = perf_counter()

            #Generar batches
            batch, advantages = get_batch(B, shapes, gamma, train_device, free_queue, full_queue, buffers)
           
            PPO_learn(micro_model, micro_model, exp_name, batch, advantages, micro_optimizer, B, n_envs)

            step += n_envs*B*T
            end = perf_counter()

            print(f"Update took {end - start}s\n-------")



        # END BATCH AND LEARN

    
    batch_and_learn(2, 100000000, exp_name)

    
    # Thread learners
    """threads = []
    for i in range(n_learner_threads):
        print("Max steps / n_threads:", max_steps // n_learner_threads)
        thread = threading.Thread(target=batch_and_learn, args=(i, max_steps // n_learner_threads,))
        thread.start()
        threads.append(thread)"""

    while True:
        k = 0
    print("TIME OUT!")


def test():
    print("Testing...")


def main():
    if args.test:
        test()
    else:
        train(args.exp_name)

if __name__ == "__main__":
    main()
