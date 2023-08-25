# Script to add the histopathological labels to a config file for end-to-end classification

import yaml
from pathlib import Path
import click

@click.command()
@click.option('--config_file', type=Path)
@click.option('--label_file', type=Path)
@click.option('--out_file', type=Path)
@click.option('--clam', type=bool, default=True)
@click.option('--target', type=str, default="fib")
def main(config_file, label_file, out_file, clam, target):
    with open(config_file, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
        
    with open(label_file, 'r') as stream:
        labelList = stream.read().splitlines()
        
    for i, line in enumerate(labelList):
        labels = line.split(';')
        tag = labels[0]

        if 'Studienr' not in tag:
            bal = int(labels[4]) # + 1 
            fib = int(labels[6]) # + 1
            sta = int(labels[1]) # + 1
            inf = int(labels[2]) # + 1
            saf = int(labels[5])
            nas = int(labels[8])
            rnas = int(labels[9])
            
            true_labels = []
            
            if "bal" in target:
                if bal == 2:
                    bal = 1
                true_labels.append(bal)
            if "fib" in target:
                if fib == 2:
                    fib = 1
                if fib == 3:
                    fib = 2
                if fib == 4:
                    fib = 2

                true_labels.append(fib)
            if "sta" in target:
                if sta == 0:
                    sta = -1
                true_labels.append(sta)
            if "inf" in target:
                true_labels.append(inf)
            if "saf" in target:
                true_labels.append(saf)
            if "nas" in target and not "rnas" in target:
                true_labels.append(nas)
            if "rnas" in target:
                true_labels.append(rnas)
            
            # true_labels = [bal, fib, sta, inf]
        
            j, dataset = find_match(data_loaded, tag)
            if j is not None:
                if clam:
                    if not (("nas" in target and not "rnas" in target) or "sta" in target or "fib" in target):
                        true_labels = [i + 1 for i in true_labels]
                    data_loaded['data'][dataset]['default'][j]['labels'] = true_labels
                else:
                    data_loaded['data'][dataset]['default'][j]['path_scores'] = true_labels
            
    with open(out_file, 'w') as file:
        yaml.dump(data_loaded, file)

def find_match(data_loaded, label_tag):
    for dataset in ['training', 'validation', 'testing']:
        for j in range(len(data_loaded['data'][dataset]['default'])):
            if label_tag in data_loaded['data'][dataset]['default'][j]['image']:
                return j, dataset
    print(label_tag)
    return None, None

if __name__ == '__main__':
    main()