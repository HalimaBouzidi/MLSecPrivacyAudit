import torch
import torch.nn as nn
import torchvision 
import numpy as np

from opacus.validators import ModuleValidator
from privacy_meter.model import PytorchModelTensor
from privacy_meter.dataset import Dataset
from privacy_meter.information_source import InformationSource

from privacy_meter.audit import Audit
from privacy_meter.metric import PopulationMetric, ReferenceMetric, ShadowMetric
from privacy_meter.information_source_signal import ModelGradientNorm, ModelLoss, ModelLogits, ModelNegativeRescaledLogits
from privacy_meter.hypothesis_test import threshold_func, linear_itp_threshold_func, logit_rescale_threshold_func, \
                                          gaussian_threshold_func, min_linear_logit_threshold_func
from privacy_meter.constants import InferenceGame
from privacy_meter.dataset import Dataset
from privacy_meter.information_source import InformationSource
from privacy_meter.model import PytorchModelTensor
from torch.utils.data import TensorDataset

from .utils import train, test

signals = {'loss': ModelLoss(), 'gradient': ModelGradientNorm(), 'logits': ModelLogits(), 'scaled_logits': ModelNegativeRescaledLogits()}

infer_games = {'privacy_loss_model': InferenceGame.PRIVACY_LOSS_MODEL}

hypo_tests = {'direct': threshold_func, 'linear_itp': linear_itp_threshold_func, 'logit_rescale': logit_rescale_threshold_func, \
              'gaussian': gaussian_threshold_func, 'min_linear_gaussian': min_linear_logit_threshold_func}

def prepare_dataset_population(train_dataset, test_dataset, train_size, test_size, population_size):
    
    all_features = np.concatenate([train_dataset.data, test_dataset.data], axis=0)
    all_targets = np.concatenate([train_dataset.targets, test_dataset.targets], axis=0)

    train_dataset.data = all_features
    train_dataset.targets = all_targets

    all_index = np.arange(len(train_dataset))
    train_index = np.random.choice(all_index, train_size, replace=False)
    test_index = np.random.choice([i for i in all_index if i not in train_index], test_size, replace=False)
    population_index = np.random.choice([i for i in all_index if i not in train_index and i not in test_index], population_size, replace=False)

    return train_dataset, test_dataset, all_index, train_index, test_index, population_index

def prepare_dataset_reference(train_dataset, test_dataset, num_reference_models, train_split, test_split):
    
    dataset = Dataset(data_dict={'train': {'x': train_dataset.data, 'y': train_dataset.targets}, 
                                 'test': {'x': test_dataset.data, 'y': test_dataset.targets}}, default_input='x',default_output='y')
    
    datasets_list = dataset.subdivide(num_splits=num_reference_models + 1, delete_original=True, in_place=False, return_results=True, method='hybrid', 
                                      split_size={'train': train_split, 'test': test_split})

    return dataset, datasets_list

def prepare_dataset_shadow(train_dataset, test_dataset, num_reference_models, split_size):
    
    dataset = Dataset(data_dict={'train': {'x': train_dataset.data, 'y': train_dataset.targets}, 
                                 'test': {'x': test_dataset.data, 'y': test_dataset.targets}}, default_input='x',default_output='y')
    
    datasets_list = dataset.subdivide(num_splits=num_reference_models + 1, delete_original=True, in_place=False, return_results=True, method='hybrid', 
                                      split_size=split_size)

    return datasets_list

def get_dataset_subset(dataset: torchvision.datasets, index):
    assert max(index) < len(dataset) and min(index) >= 0
    data = (torch.from_numpy(dataset.data[index]).float().permute(0, 3, 1, 2) / 255) 
    targets = list(np.array(dataset.targets)[index])
    targets = torch.tensor(targets, dtype=torch.long)
    return data, targets

def get_target_audit_population(train_dataset, train_index, test_index, population_index):
    
    train_data, train_targets = get_dataset_subset(train_dataset, train_index)
    test_data, test_targets = get_dataset_subset(train_dataset, test_index)
    audit_data, audit_targets = get_dataset_subset(train_dataset, population_index)
    
    target_dataset = Dataset( data_dict={"train": {"x": train_data, "y": train_targets}, 
                                         "test": {"x": test_data, "y": test_targets},},
                                         default_input="x", default_output="y",)
    
    audit_dataset = Dataset( data_dict={"train": {"x": audit_data, "y": audit_targets}},
                                        default_input="x", default_output="y",)
    
    return target_dataset, audit_dataset
    

def get_subset_dataloader(args, train_dataset, train_index, test_index):
    
    train_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(train_dataset,train_index),
            batch_size=args['data']['batch_size'],
            shuffle=True,
            num_workers=args['data']['data_loader_workers_per_gpu'],
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=16)
    
    test_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(train_dataset,test_index),
            batch_size=args['data']['batch_size'],
            shuffle=True,
            num_workers=args['data']['data_loader_workers_per_gpu'],
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=16)
    
    return train_loader, test_loader
    
def get_full_dataloader(args, train_dataset, test_dataset):
    
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args['data']['batch_size'],
            shuffle=True,
            num_workers=args['data']['data_loader_workers_per_gpu'], 
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=16)
    
    test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args['data']['batch_size'],
            shuffle=True,
            num_workers=args['data']['data_loader_workers_per_gpu'],
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=16)
    
    return train_loader, test_loader


def population_attack(args, model, train_dataset, test_dataset, device):
    
    train_size, test_size, population_size = args['attack']['train_size'], args['attack']['test_size'], args['attack']['population_size']

    train_dataset, test_dataset, all_index, train_index, test_index, population_index = \
                   prepare_dataset_population(train_dataset, test_dataset, train_size, test_size, population_size)
    
    train_loader, test_loader = get_subset_dataloader(args, train_dataset, train_index, test_index)
    
    criterion = nn.CrossEntropyLoss()
    model = train(model, args['train']['epochs'], args['train']['optimizer'], criterion, train_loader, test_loader, len(train_index), len(test_index), device)
    test_loss, test_accuracy = test(model, test_loader, len(test_index), device, criterion)
    
    ModuleValidator.fix(model)
    target_model = PytorchModelTensor(model_obj=model, loss_fn=criterion, device=device,batch_size=10)
    target_dataset, audit_dataset = get_target_audit_population(train_dataset, train_index, test_index, population_index)

    target_info_source = InformationSource(models=[target_model], datasets=[target_dataset])
    reference_info_source = InformationSource(models=[target_model], datasets=[audit_dataset])
    
    metric = PopulationMetric(target_info_source=target_info_source, reference_info_source=reference_info_source,
                              signals=[signals[args['attack']['signal']]], hypothesis_test_func=hypo_tests[args['attack']['hypo_test']])
    
    audit_obj = Audit(metrics=metric, inference_game_type=infer_games[args['attack']['privacy_game']], target_info_sources=target_info_source,
                      reference_info_sources=reference_info_source)
    
    audit_obj.prepare()
    audit_results = audit_obj.run()[0]
    
    return audit_results, test_accuracy
    
# This is the problem      
def reference_attack(args, model, train_dataset, test_dataset, device):
    
    n_ref_models, train_split, test_split = args['attack']['n_ref_models'], args['attack']['train_size'], args['attack']['test_size']
    fpr_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    datasets_list = prepare_dataset_reference(train_dataset, test_dataset, n_ref_models, train_split, test_split)

    train_set = TensorDataset(torch.Tensor(datasets_list[0].get_feature('train', '<default_input>')), \
                              torch.Tensor(datasets_list[0].get_feature('train', '<default_output>')))
    
    test_set = TensorDataset(torch.Tensor(datasets_list[0].get_feature('train', '<default_input>')), \
                              torch.Tensor(datasets_list[0].get_feature('train', '<default_output>')))
    
    train_loader, test_loader = get_full_dataloader(args, train_set, test_set)
    
    exit()

    criterion = nn.CrossEntropyLoss()
    model = train(model, args['train']['epochs'], args['train']['optimizer'], criterion, train_loader, test_loader, train_split, test_split, device)
    
    ModuleValidator.fix(model)
    target_model = PytorchModelTensor(model_obj=model, loss_fn=criterion, device=device,batch_size=10)

    reference_models = []
    for model_idx in range(n_ref_models):
        reference_model = model.deep_copy()
        ref_train_set = TensorDataset(torch.Tensor(datasets_list[model_idx].get_feature('train', '<default_input>')), \
                              torch.Tensor(datasets_list[model_idx].get_feature('train', '<default_output>')))
    
        ref_test_set = TensorDataset(torch.Tensor(datasets_list[model_idx].get_feature('train', '<default_input>')), \
                                torch.Tensor(datasets_list[model_idx].get_feature('train', '<default_output>')))
        
        ref_train_loader, ref_test_loader = get_full_dataloader(args, ref_train_set, ref_test_set)
        
        reference_model = train(model, args['train']['epochs'], args['train']['optimizer'], criterion, ref_train_loader, ref_test_loader, train_split, test_split, device)
        reference_models.append(PytorchModelTensor(model_obj=reference_model, loss_fn=criterion))

    target_info_source = InformationSource(models=[target_model], datasets=[datasets_list[0]])
    reference_info_source = InformationSource(models=reference_models, datasets=[datasets_list[0]])
    
    metric = ReferenceMetric(target_info_source=target_info_source, reference_info_source=reference_info_source,
                             signals=[signals[args['attack']['signal']]], hypothesis_test_func=hypo_tests[args['attack']['hypo_test']])
    
    audit_obj = Audit(metrics=metric, inference_game_type=infer_games[args['attack']['privacy_game']], target_info_sources=target_info_source,
                      reference_info_sources=reference_info_source, fpr_tolerances=fpr_list)
    
    audit_obj.prepare()
    audit_results = audit_obj.run()[0]

    return audit_results


def shadow_attack(args, model, train_dataset, test_dataset, device):
    
    n_shadow_models, split_size = args['attack']['n_shadow_models'], args['attack']['split_size']
    shadow_models = [model.deep_copy() for _ in range(n_shadow_models)] 
    
    dataset, datasets_list = prepare_dataset_shadow(train_dataset, test_dataset, shadow_models, split_size)
    
    criterion = nn.CrossEntropyLoss()
    for model_idx in range(len(shadow_models)):
        x = dataset.get_feature(split_name=f'train{model_idx:03d}', feature_name='<default_input>')
        y = dataset.get_feature(split_name=f'train{model_idx:03d}', feature_name='<default_output>')
        ref_train_set = TensorDataset(torch.Tensor(x), torch.Tensor(y))
    
        x = dataset.get_feature(split_name=f'train{model_idx:03d}', feature_name='<default_input>')
        y = dataset.get_feature(split_name=f'train{model_idx:03d}', feature_name='<default_output>')
        ref_test_set = TensorDataset(torch.Tensor(x), torch.Tensor(y))

        ref_train_loader, ref_test_loader = get_full_dataloader(args, ref_train_set, ref_test_set)
        
        shadow_model = train(shadow_models[model_idx], args['train']['epochs'], args['train']['optimizer'], criterion, ref_train_loader, ref_test_loader, split_size, split_size, device)
        shadow_models[model_idx].append(PytorchModelTensor(model_obj=shadow_model, loss_fn=criterion))

    target_info_source = InformationSource(models=[shadow_models[0]], datasets=[datasets_list[0]])
    
    reference_info_source = InformationSource(models=shadow_models[1:], datasets=datasets_list[1:])
    
    metric = ShadowMetric(target_info_source=target_info_source, reference_info_source=reference_info_source, signals=[signals[args['attack']['signal']]], \
                          hypothesis_test_func=hypo_tests[args['attack']['hypo_test']], unique_dataset=False, reweight_samples=True)
    
    audit_obj = Audit(metrics=metric, inference_game_type=infer_games[args['attack']['privacy_game']], target_info_sources=target_info_source,
                      reference_info_sources=reference_info_source)
    
    audit_results = audit_obj.prepare()

    return audit_results
