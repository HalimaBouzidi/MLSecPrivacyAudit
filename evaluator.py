import argparse
import os, random
import csv, torch, yaml
import torch.nn as nn
import torch.optim as optim

from data.data_loader import build_datasets
from models.models import get_model
from attacks.member_inference import population_attack, reference_attack, shadow_attack
from attacks.utils import train, test

TF_ENABLE_ONEDNN_OPTS=0

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--cf", type=str, default="./configs/population_evaluate.yaml", help="Yaml file which contains the configurations")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the experiments")

    # Load the parameters
    args = parser.parse_args()
    with open(args.cf, "rb") as f:
        configs = yaml.load(f, Loader=yaml.Loader)

    print('\n ************************************************************************************************ \n')

    train_dataset, val_dataset = build_datasets(configs)

