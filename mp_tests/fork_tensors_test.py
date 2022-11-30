import torch
import numpy as np
import multiprocessing as mp


def check_tensor(x: torch.Tensor, parent: bool=False):
    """ Muestra por consola el tensor x y muestra quien lo llama a verificar """
    
    if parent:
        print("Tensor en padre:", x, "   id:", id(x))
    else:
        print("Tensor en hijo:", x, "   id:", id(x))


def modify_tensor(x: torch.Tensor, pos: int, val: int):
    """ Modifica el tensor x[pos] = val """

    x[pos] = val
    check_tensor(x)



def main() -> int:

    # Obtener contexto
    ctx = mp.get_context("spawn")

    # Crear tensor
    x = torch.from_numpy(np.array([1, 2, 3]))
    x.share_memory_()

    # Verificar tensor antes de modificarse en proceso hijo
    check_tensor(x, True)

    # Modificar tensor en proceso hijo con Spawn
    actor_process = ctx.Process(target=modify_tensor, args=(x, 1, 10))
    actor_process.start()

    # Esperar a que el proceso hijo termine
    actor_process.join()

    # Verificar cambios en el tensor
    check_tensor(x, True)


if __name__ == "__main__":
    main()


    


