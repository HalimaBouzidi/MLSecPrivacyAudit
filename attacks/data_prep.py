import torch, torchvision
import numpy as np
from privacy_meter.dataset import Dataset


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

def prepare_dataset_reference(dataset, num_reference_models, train_split, test_split):
    
    datasets_list = dataset.subdivide(num_splits=num_reference_models + 1, delete_original=True, in_place=False, return_results=True, method='hybrid', 
                                      split_size={'train': train_split, 'test': test_split})
    
    return datasets_list

def prepare_dataset_shadow(dataset, n_shadow_models, split_size):

    datasets_list = dataset.subdivide(num_splits=n_shadow_models + 1, delete_original=True, in_place=True, return_results=True,
                                      method='hybrid', split_size=split_size)

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
            drop_last = True,
            prefetch_factor=16)
    
    test_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(train_dataset,test_index),
            batch_size=args['data']['batch_size'],
            shuffle=True,
            num_workers=args['data']['data_loader_workers_per_gpu'],
            pin_memory=True,
            persistent_workers=True,
            drop_last = True,
            prefetch_factor=16)
    
    return train_loader, test_loader
    
def get_dataset_full(dataset: torchvision.datasets):
    data = (torch.from_numpy(dataset.data).float().permute(0, 3, 1, 2) / 255) 
    targets = list(np.array(dataset.targets))
    targets = torch.tensor(targets, dtype=torch.long)
    return data, targets

def get_target_reference(train_dataset, test_dataset):
    
    train_data, train_targets = get_dataset_full(train_dataset)
    test_data, test_targets = get_dataset_full(test_dataset)

    target_dataset = Dataset( data_dict={"train": {"x": train_data, "y": train_targets}, 
                                         "test": {"x": test_data, "y": test_targets},},
                                         default_input="x", default_output="y",)
    
    return target_dataset

def get_full_dataloader(args, train_dataset, test_dataset):
    
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args['data']['batch_size'],
            shuffle=True,
            num_workers=args['data']['data_loader_workers_per_gpu'], 
            pin_memory=True,
            persistent_workers=True,
            drop_last = True,
            prefetch_factor=16)
    
    test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args['data']['batch_size'],
            shuffle=True,
            num_workers=args['data']['data_loader_workers_per_gpu'],
            pin_memory=True,
            persistent_workers=True,
            drop_last = True,
            prefetch_factor=16)
    
    return train_loader, test_loader