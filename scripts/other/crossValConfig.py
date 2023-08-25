# Script to create multiple config files for crossvalidation

import click
import yaml
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import sms
import copy


@click.command()
@click.option('--config_file', type=Path)
@click.option('--kfolds', type=int, default=4)
@click.option('--test', type=bool, default=True)
@click.option('--out_regex', type=str)
def main(config_file, kfolds, out_regex, test):
    skf = StratifiedKFold(n_splits=kfolds)

    X, y, data_loaded = parse_config(config_file=config_file, test=test)

    for i, (train, test) in enumerate(skf.split(X, y)):

        train_set = [X[j] for j in train]
        test_set = [X[j] for j in test]
        write_config(train_set, test_set, out_regex.format(fold=str(i)), copy.deepcopy(data_loaded), test=test)

def parse_config(config_file, test):
    
    with open(config_file, 'r') as stream:
        data_loaded = yaml.safe_load(stream)

    # test_data = data_loaded['data']['testing']['default']
    data = data_loaded['data']['training']['default']
    data += data_loaded['data']['validation']['default']
    if test:
        data += data_loaded['data']['testing']['default']

    labels = [i['labels'] for i in data]

    # meta_data = {'distribuition':data['data']['distribution'], 
    #              'path':data['data']['path'], 
    #              'type':data['data']['type']}

    
    return data, labels, data_loaded

def write_config(train_set, test_set, out_file, data_loaded, test):
    Path(out_file).parents[0].mkdir(parents=True, exist_ok=True)

    data_loaded['data']['training']['default'] = train_set
    if test:
        labels = [i['labels'] for i in test_set]
        val_set, test_set, _, _ = sms.train_test_split(test_set, labels, stratify=labels, test_size=10)

        data_loaded['data']['validation']['default'] = val_set
        data_loaded['data']['testing']['default'] = test_set
    else:
        data_loaded['data']['validation']['default'] = test_set

    with open(out_file, 'w') as file:
        yaml.dump(data_loaded, file)

if __name__ == "__main__":
    main()