import argparse
import os, random
import csv, torch, yaml
import torch.nn as nn
import torch.optim as optim
import numpy as np

from data.data_loader import build_datasets
from models.models import get_model
from attacks.member_inference import population_attack, reference_attack, shadow_attack
from attacks.utils import train, test

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--cf", type=str, default="./configs/population_evaluate.yaml", help="Yaml file which contains the configurations")

    args = parser.parse_args()
    with open(args.cf, "rb") as f:
        configs = yaml.load(f, Loader=yaml.Loader)

    seed_value = configs['run']['seed']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('\n ******************************** START of THE SCRIPT **************************************************** \n')

    train_dataset, test_dataset = build_datasets(configs)

    model = get_model(configs)
    model.to(device)

    if configs['attack']['type'] == 'population':
        audit_results, test_accuracy = population_attack(configs, model, train_dataset, test_dataset, device)

    elif configs['attack']['type'] == 'reference':
        reference_attack(configs, model, train_dataset, test_dataset, device)

    elif configs['attack']['type'] == 'shadow':
        shadow_attack(configs, model, train_dataset, test_dataset, device)

    else:
        raise NotImplementedError(f"{configs['attack']['type']} is not implemented")