import multiprocessing as mp
import torch
import time

# Some aliases
Queue = mp.SimpleQueue


def foo(x: torch.Tensor):
    for i in range(x.size()[0]):
        x[i] += 2

def foo2(x: list):
    for i in range(len(x)):
        x[i] += 2
    print("Lista en subproceso:", x)


def queue(queue: Queue, elements: list):
    for x in elements:
        queue.put(x)
        time.sleep(1)

def dequeue(queue: Queue, n: torch.Tensor):
    # Mientras el contador no sea 0, dequeue
    while n[0] != 0:
        # Si cola no vacia, dequeue
        if not queue.empty():
            print("Dequeue:", queue.get())
            n[0] -= 1






if __name__ == '__main__':

    # --- Compartir un tensor entre procesos --- #
    print("\nTest de tensores compartidos entre procesos")
    my_tensor = torch.Tensor([1, 2, 3])
    my_tensor.share_memory_()
    ctx = mp.get_context("spawn")

    p1 = ctx.Process(target=foo, args=(my_tensor,))
    p1.start()
    p1.join()

    print(my_tensor)


    # --- Cuantos cores tengo disponible en mi cpu? --- #
    print("\nCuantos cores tengo en mi CPU?")
    print(mp.cpu_count())

    # --- Crear queues que compartan info entre procesos --- #
    print("\nTest de Queue entre procesos...")
    n = torch.Tensor([10])    # Esto servirá de contador para dequeue
    my_queue = ctx.SimpleQueue()

    queuer_1 = ctx.Process(target=queue, args=(my_queue, [1, 2, 3, 4, 5],))
    queuer_2 = ctx.Process(target=queue, args=(my_queue, [6, 7, 8, 9, 10],))
    dequeuer = ctx.Process(target=dequeue, args=(my_queue, n))

    queuer_1.start()
    queuer_2.start()
    dequeuer.start()

    queuer_1.join()
    queuer_2.join()
    dequeuer.join()

    # --- Se comparte una lista con fork si lo doy como argumento?
    print("\nSe comparte una lista en fork si la doy como argumento?")
    lista = [1, 2, 3, 4]
    ctx = mp.get_context("fork")
    p1 = ctx.Process(target=foo2, args=(lista,))
    p1.start()
    p1.join()
    print("Lista en main:", lista)
    # Respuesta: NO, es una copia
