import argparse
import random
import torch, yaml, csv
import numpy as np
import torch.nn as nn

from data.data_loader import build_datasets, build_data_loader
from models.basics import get_model
from attacks.utils import train, test

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="searchable_alexnet", help="Model to evaluate")
    parser.add_argument("--width", type=float, default=1.0, help="Width expand ratio")
    parser.add_argument("--depth", type=float, default=1.0, help="Number of model layers")

    args = parser.parse_args()
    cf = "./configs/train_config.yaml"
    
    with open(cf, "rb") as f:
        configs = yaml.load(f, Loader=yaml.Loader)

    configs['train']['model_name'] = args.model
    configs['train']['width_multi'] = args.width
    configs['train']['depth_multi'] = args.depth
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_value = configs['run']['seed']
    set_seed(seed_value)

    model = get_model(configs)
    model.to(device)

    train_dataset, test_dataset = build_datasets(configs)
    train_loader, test_loader, train_sampler = build_data_loader(configs)

    criterion, path = nn.CrossEntropyLoss(), configs['run']['saved_models']
    lr, w_decay = float(configs['train']['learning_rate']), float(configs['train']['weight_decay'])
    model = train(model, configs['train']['epochs'], configs['train']['optimizer'], criterion, lr, w_decay, train_loader, test_loader, device, path)
    test_loss, test_accuracy = test(model, test_loader, device, criterion)

    print("************ TEST ACCURACY ***************", test_accuracy)

    with open('./target_accuracy.csv', 'a') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow([configs['train']['model_name'], configs['train']['width_multi'], configs['train']['depth_multi'], test_accuracy])

    