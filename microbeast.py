import numpy as np
import multiprocessing
import torch
from parser import *
import typing

from libs.utils import create_buffers, create_env

# TO DO: Crear lista para append los actores
# TO DO: Crear modelo learner y Optimizer
# TO DO: Llenar Free queue y Full queue
# TO DO: Funcion Act() donde se crea un Agente nuevo y una NN nueva para entrenar

# Un Buffer será un diccionario con keys string y claves list(Tensor)

def train():
    print("Training...")
    # Crear ambiente para obtener shapes
    micro_env = create_env(8, 2, 512) 

    # Crear buffer
    create_buffers(4, 2, 80, micro_env.observation_space.shape)
    # Ya no sirve "micro_envs", deshechar...

    # ----- Crear Queues ----- #
    # free_queue para indices de buffers disponibles para llenar
    # full_queue para indices de buffers con trayectorias para backprop
    ctx = mp.get_context("spawn")
    free_queue = ctx.SimpleQueue()
    full_queue = ctx.SimpleQueue()

    # ----- Crear subprocesos (Actores) ----- #
    actor_processes = []

def test():
    print("Testing...")


def main():
    if args.test:
        test()
    else:
        train()

if __name__ == "__main__":
    main()
