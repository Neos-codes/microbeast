import argparse

# ---- Parsear argumentos
parser = argparse.ArgumentParser(description="Microbeast: Gym-microRTS, Impala and PPO option flags")

# Modo: Train by default, Test using this flag
parser.add_argument("--test", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="Use this flag if you want to test a model")

# Nombre de experimento
parser.add_argument("--exp_name", type=str, default="No_name", nargs="?", help="Name of the result table of this experiment")

# Procesas argumentos
args = parser.parse_args()


