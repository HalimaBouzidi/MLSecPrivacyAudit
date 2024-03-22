import torch, copy
import torch.nn as nn

from opacus.validators import ModuleValidator
from privacy_meter.model import PytorchModelTensor
from privacy_meter.information_source import InformationSource

from privacy_meter.audit import Audit
from privacy_meter.metric import PopulationMetric, ReferenceMetric, ShadowMetric
from privacy_meter.information_source_signal import ModelGradientNorm, ModelLoss, ModelLogits, ModelNegativeRescaledLogits
from privacy_meter.hypothesis_test import threshold_func, linear_itp_threshold_func, logit_rescale_threshold_func, \
                                          gaussian_threshold_func, min_linear_logit_threshold_func

from privacy_meter.constants import InferenceGame
from privacy_meter.information_source import InformationSource
from privacy_meter.model import PytorchModelTensor
from torch.utils.data import TensorDataset

from .utils import train, test
from .data_prep import *

from privacy_meter.audit_report import ROCCurveReport

signals = {'loss': ModelLoss(), 'gradient': ModelGradientNorm(), 'logits': ModelLogits(), 'scaled_logits': ModelNegativeRescaledLogits()}

infer_games = {'privacy_loss_model': InferenceGame.PRIVACY_LOSS_MODEL}

hypo_tests = {'direct': threshold_func, 'linear_itp': linear_itp_threshold_func, 'logit_rescale': logit_rescale_threshold_func, \
              'gaussian': gaussian_threshold_func, 'min_linear_gaussian': min_linear_logit_threshold_func}

def population_attack(args, model, train_dataset, test_dataset, device):
    
    train_size, test_size, population_size = args['attack']['train_size'], args['attack']['test_size'], args['attack']['population_size']

    train_dataset, test_dataset, all_index, train_index, test_index, population_index = \
                   prepare_dataset_population(train_dataset, test_dataset, train_size, test_size, population_size)
    
    train_loader, test_loader = get_subset_dataloader(args, train_dataset, train_index, test_index)
    
    criterion, path = nn.CrossEntropyLoss(), args['run']['saved_models']
    lr, w_decay = float(args['train']['learning_rate']), float(args['train']['weight_decay'])
    model = train(model, args['train']['epochs'], args['train']['optimizer'], criterion, lr, w_decay, train_loader, test_loader, device, path)
    test_loss, test_accuracy = test(model, test_loader, device, criterion)
    print('************ TEST ACCURACY: ', test_accuracy)

    ModuleValidator.fix(model)
    target_model = PytorchModelTensor(model_obj=model, loss_fn=criterion, device=device,batch_size=args['data']['batch_size'])
    target_dataset, audit_dataset = get_target_audit_population(train_dataset, train_index, test_index, population_index)

    target_info_source = InformationSource(models=[target_model], datasets=[target_dataset])
    reference_info_source = InformationSource(models=[target_model], datasets=[audit_dataset])
    
    metric = PopulationMetric(target_info_source=target_info_source, reference_info_source=reference_info_source,
                              signals=[signals[args['attack']['signal']]], hypothesis_test_func=hypo_tests[args['attack']['hypo_test']])
    
    log_dir = args['attack']['log_dir']+'/population_'+args['attack']['test_name']
    audit_obj = Audit(metrics=metric, inference_game_type=infer_games[args['attack']['privacy_game']], target_info_sources=target_info_source,
                      reference_info_sources=reference_info_source, logs_directory_names= [log_dir, log_dir, log_dir])
    
    audit_obj.prepare()
    audit_results = audit_obj.run()[0]
    infer_game = infer_games[args['attack']['privacy_game']]
    
    return audit_results, infer_game, test_accuracy
    
# This is the problem      
def reference_attack(args, model, train_dataset, test_dataset, device):
    
    n_ref_models, train_split, test_split = args['attack']['n_ref_models'], args['attack']['train_size'], args['attack']['test_size']
    fpr_list = [0.0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    dataset = get_target_reference(train_dataset, test_dataset)

    datasets_list = prepare_dataset_reference(dataset, n_ref_models, train_split, test_split)

    train_set = TensorDataset(torch.Tensor(datasets_list[0].get_feature('train', '<default_input>')), \
                              torch.Tensor(datasets_list[0].get_feature('train', '<default_output>')))
    
    test_set = TensorDataset(torch.Tensor(datasets_list[0].get_feature('test', '<default_input>')), \
                              torch.Tensor(datasets_list[0].get_feature('test', '<default_output>')))
    
    train_loader, test_loader = get_full_dataloader(args, train_set, test_set)
    
    orig_model = copy.deepcopy(model)
    
    criterion, path = nn.CrossEntropyLoss(), args['run']['saved_models']
    lr, w_decay = float(args['train']['learning_rate']), float(args['train']['weight_decay'])
    model = train(model, args['train']['epochs'], args['train']['optimizer'], criterion, lr, w_decay, train_loader, test_loader, device, path)
    test_loss, test_accuracy = test(model, test_loader, device, criterion)
    print('************ TEST ACCURACY: ', test_accuracy)

    ModuleValidator.fix(model)
    target_model = PytorchModelTensor(model_obj=model, loss_fn=criterion, device=device,batch_size=args['data']['batch_size'])

    reference_models = []
    for model_idx in range(n_ref_models):
        reference_model = copy.deepcopy(orig_model)
        ref_train_set = TensorDataset(torch.Tensor(datasets_list[model_idx].get_feature('train', '<default_input>')), \
                              torch.Tensor(datasets_list[model_idx].get_feature('train', '<default_output>')))
    
        ref_test_set = TensorDataset(torch.Tensor(datasets_list[model_idx].get_feature('test', '<default_input>')), \
                                torch.Tensor(datasets_list[model_idx].get_feature('test', '<default_output>')))
        
        ref_train_loader, ref_test_loader = get_full_dataloader(args, ref_train_set, ref_test_set)
        
        reference_model = train(reference_model, args['train']['epochs'], args['train']['optimizer'], criterion, lr, w_decay, \
                                ref_train_loader, ref_test_loader, device, path)
        
        reference_models.append(PytorchModelTensor(model_obj=reference_model, loss_fn=criterion))
        
    target_info_source = InformationSource(models=[target_model], datasets=[datasets_list[0]])
    reference_info_source = InformationSource(models=reference_models, datasets=[datasets_list[0]])
    
    metric = ReferenceMetric(target_info_source=target_info_source, reference_info_source=reference_info_source,
                             signals=[signals[args['attack']['signal']]], hypothesis_test_func=hypo_tests[args['attack']['hypo_test']])

    log_dir = args['attack']['log_dir']+'/reference_'+args['attack']['test_name']
    audit_obj = Audit(metrics=metric, inference_game_type=infer_games[args['attack']['privacy_game']], target_info_sources=target_info_source,
                      reference_info_sources=reference_info_source, fpr_tolerances=fpr_list, logs_directory_names= [log_dir, log_dir, log_dir])
    
    audit_obj.prepare()
    audit_results = audit_obj.run()[0]
    infer_game = infer_games[args['attack']['privacy_game']]

    audit_results, infer_game = None, None

    return audit_results, infer_game, test_accuracy


def shadow_attack(args, model, train_dataset, test_dataset, device):

    n_shadow_models, split_size = args['attack']['n_shadow_models'], args['attack']['split_size']
    shadow_models = [copy.deepcopy(model) for _ in range(n_shadow_models)] 

    dataset = get_target_reference(train_dataset, test_dataset)

    datasets_list = prepare_dataset_shadow(dataset, n_shadow_models, split_size)
        
    criterion, path = nn.CrossEntropyLoss(), args['run']['saved_models']
    lr, w_decay = float(args['train']['learning_rate']), float(args['train']['weight_decay'])
    trained_shadow_models = []
    
    for model_idx in range(len(shadow_models)):
        x = dataset.get_feature(split_name=f'train{model_idx:03d}', feature_name='<default_input>')
        y = dataset.get_feature(split_name=f'train{model_idx:03d}', feature_name='<default_output>')
        ref_train_set = TensorDataset(torch.Tensor(x), torch.Tensor(y))
    
        x = dataset.get_feature(split_name=f'test{model_idx:03d}', feature_name='<default_input>')
        y = dataset.get_feature(split_name=f'test{model_idx:03d}', feature_name='<default_output>')
        ref_test_set = TensorDataset(torch.Tensor(x), torch.Tensor(y))

        ref_train_loader, ref_test_loader = get_full_dataloader(args, ref_train_set, ref_test_set)
        
        shadow_model = train(shadow_models[model_idx], args['train']['epochs'], args['train']['optimizer'], \
                             criterion, lr, w_decay, ref_train_loader, ref_test_loader, device, path)
        
        if model_idx == 0:
            test_loss, test_accuracy = test(shadow_model, ref_test_loader, device, criterion)
            print('************ TEST ACCURACY: ', test_accuracy)
        
        trained_shadow_models.append(PytorchModelTensor(model_obj=shadow_model, loss_fn=criterion))


    target_info_source = InformationSource(models=[trained_shadow_models[0]], datasets=[datasets_list[0]])
    
    reference_info_source = InformationSource(models=trained_shadow_models[1:], datasets=datasets_list[1:])
    
    metric = ShadowMetric(target_info_source=target_info_source, reference_info_source=reference_info_source, signals=[signals[args['attack']['signal']]], \
                          hypothesis_test_func=hypo_tests[args['attack']['hypo_test']], unique_dataset=False, reweight_samples=True)
    
    log_dir = args['attack']['log_dir']+'/shadow_'+args['attack']['test_name']
    audit_obj = Audit(metrics=metric, inference_game_type=infer_games[args['attack']['privacy_game']], target_info_sources=target_info_source,
                      reference_info_sources=reference_info_source, logs_directory_names= [log_dir, log_dir, log_dir])
        
    audit_obj.prepare()
    audit_results = audit_obj.run()[0]
    infer_game = infer_games[args['attack']['privacy_game']]

    return [[audit_results]], infer_game, test_accuracy
