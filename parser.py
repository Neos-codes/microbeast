import argparse

# ---- Parsear argumentos
parser = argparse.ArgumentParser(description="Microbeast: Gym-microRTS, Impala and PPO option flags")

# Modo: Train by default, Test using this flag
parser.add_argument("--test", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="Use this flag if you want to test a model")


# Procesas argumentos
args = parser.parse_args()


