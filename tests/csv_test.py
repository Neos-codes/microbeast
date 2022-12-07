import csv
import multiprocessing as mp
import numpy as np


def print_csv(file_name: str, data: list):
    """ Añade al final de un csv una lista de datos """

    with open(file_name, "a") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(data)
        
        csv_file.close()


def par(file_name: str, n_rows: int, n_columns: int):
    """ Añade al final de un archivo csv una lista de numeros pares """
    
    data = []
    for i in range(n_rows):
        for j in range(n_columns):
            data.append(2*j)
        print_csv(file_name, data)
        data.clear()


def impar(file_name: str, n_rows: int, n_columns: int):
    """ Añade al final de un archivo csv una lista de numeros impares """

    data = []
    for i in range(n_rows):
        for j in range(n_columns):
            data.append(2*j + 1)
        print_csv(file_name, data)
        data.clear()


def main() -> int:

    # Crear archivo .csv
    with open("test.csv", "w") as data:
        pass

    # Obtener contexto
    ctx = mp.get_context("spawn")

    # Crear procesos
    processes = []

    # Proceso de pares
    actor = ctx.Process(target=par, args=("test.csv", 1000, 5))
    processes.append(actor)
    actor.start()

    # Proceso de impares
    actor = ctx.Process(target=impar, args=("test.csv", 1000, 5))
    processes.append(actor)
    actor.start()


    for i in range(2):
        processes[i].join()


if __name__ == "__main__":
    main()
