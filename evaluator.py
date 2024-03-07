import argparse
import random
import torch, yaml
import numpy as np

from data.data_loader import build_datasets
from models.models import get_model
from attacks.member_inference import population_attack, reference_attack, shadow_attack
from analyzer.plot import * 
from privacy_meter.audit_report import ROCCurveReport, SignalHistogramReport

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", type=str, default="population", help="Type of MIA attack", choices=["population", "reference", "shadow"])
    parser.add_argument("--plot", type=str, default="True", help="Wheter to visualize results", choices=["True", "False"])

    args = parser.parse_args()
    cf = "./configs/"+args.attack+"_attack_evaluate.yaml"
    
    with open(cf, "rb") as f:
        configs = yaml.load(f, Loader=yaml.Loader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_value = configs['run']['seed']
    set_seed(seed_value)

    print('\n ******************************** START of THE SCRIPT **************************************************** \n')

    train_dataset, test_dataset = build_datasets(configs)

    model = get_model(configs)
    model.to(device)

    if configs['attack']['type'] == 'population':
        audit_results, infer_game, test_accuracy = population_attack(configs, model, train_dataset, test_dataset, device)

    elif configs['attack']['type'] == 'reference':
        audit_results, infer_game, test_accuracy = reference_attack(configs, model, train_dataset, test_dataset, device)

    elif configs['attack']['type'] == 'shadow':
        audit_results, infer_game, test_accuracy = shadow_attack(configs, model, train_dataset, test_dataset, device)

    else:
        raise NotImplementedError(f"{configs['attack']['type']} is not implemented")
    
    
    if args.plot == 'True':
        
        logdir = configs['attack']['log_dir']+'/'+args.attack+'_'+configs['attack']['test_name']
        
        ROCCurveReport.generate_report(metric_result=audit_results,
                                       inference_game_type=infer_game, show=True, filename = logdir+'/roc_curve.png')

        if args.attack == 'shadow':            
            SignalHistogramReport.generate_report(metric_result=audit_results[0][0], inference_game_type=infer_game,
                                              show=True, filename = logdir+'/signal_histogram.png')
            
        else:
            SignalHistogramReport.generate_report(metric_result=audit_results[0], inference_game_type=infer_game,
                                                show=True, filename = logdir+'/signal_histogram.png')
            
            plot_log_scale_roc_curve(audit_results, logdir, logdir + "/log_scale_roc_curve.png")


        