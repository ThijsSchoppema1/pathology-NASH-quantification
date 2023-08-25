# Script to evaluate the MIL model produced by trainModel.py with task classificationPatch

import click
from dataProcesses import augmentations, dataGenerators
from torch.utils.data import DataLoader
import yaml
import torch
from pathlib import Path
import trainModel as tm
import copy

import json

@click.command()
@click.option('--model_path', type=Path)
@click.option('--out_file', type=Path)
@click.option('--config_file', type=Path)
@click.option('--target', type=str, default='testing')
@click.option('--batch_size', type=int, default=16)
@click.option('--n_classes', type=int, default=3)
def main(
        batch_size,
        model_path,
        out_file,
        config_file,
        target,
        n_classes
    ):
    paths_p, sampler_p, train_p, model_p, _ = tm.parse_config(config_file=config_file)
    model, loss_fn, pl_model = tm.prepare_model(model_p, sampler_p, train_p)
    
    data_dict, root, patches = parse_data_config(paths_p['source_file'], target)
    
    model.load_state_dict(torch.load(model_path))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('We are using the {} device.'.format(device))
    model = model.to(device)
    model.eval()
    
    pred_dict = {}
    for i, sample in enumerate(data_dict):
        sample_name = str(Path(sample['image'].format(root=root, images=patches)).stem)
        image_dir = Path(Path(sample['image'].format(root=root, images=patches)).with_suffix(''))
        
        data_gen = dataGenerators.imagePatchSamplerInference(image_dir)
        data_gen.step()
        
        inf_dataloader = DataLoader(data_gen, batch_size=batch_size, shuffle=False)
        
        print('process sample, {s}: {i} of {n}'.format(s=sample_name, i=i, n=len(data_dict)))
        pred_dict[sample_name] = []
        for inputs in inf_dataloader:
            inputs = inputs.to(device)
            
            output = model(inputs).detach().cpu().numpy()

            pred = output.argmax(axis=1).tolist()

            pred_dict[sample_name].append(pred)
            
        pred_dict[sample_name] = [*pred_dict[sample_name]]
        
    write_result_dict(pred_dict, out_file)    
    
    final_pred = []
    for key, value in pred_dict.items():
        counter = [key]
        for i in range(n_classes):
            counter.append(sum(row.count(i) for row in value))
        final_pred.append(copy.deepcopy(counter))
        
    write_result_csv(final_pred, out_file.with_suffix('.csv'))
        
    

def parse_data_config(config_file, target):
    with open(config_file, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    data_dict = data_loaded['data'][target]['default']
    root = data_loaded['path']['root']
    patches = data_loaded['path']['patches']
    return data_dict, root, patches

def write_result_dict(pred_dict, out_file):
    with open(out_file, 'w+') as f:
        json.dump(pred_dict, f)
        
def write_result_csv(pred_list, out_file):
    results = []
    for pred in pred_list:
        pred = [str(i) for i in pred]
        results.append(';'.join(pred))
    
    with open(out_file, 'w+') as f:
        f.write('\n'.join(results))
        

if __name__ == '__main__':
    main()