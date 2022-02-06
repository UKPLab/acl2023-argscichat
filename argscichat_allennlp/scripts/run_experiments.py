"""

Simple utility script for running multiple experiments in sequence

"""

from subprocess import Popen
from argscichat_allennlp.utility.json_utils import save_json
import simplejson as sj
import _jsonnet
import os
import argscichat_allennlp.const_define as cd
from tqdm import tqdm


def run_experiment(experiment_info):
    # Read config file
    base_config_path = os.path.join(cd.PROJECT_DIR, 'training_config')
    config_path = os.path.join(base_config_path, 'led_argscichat.jsonnet')
    config_data = sj.loads(_jsonnet.evaluate_file(config_path))

    # Update config file
    experiment_params = experiment_info['params']

    # Data paths
    base_data_path = os.path.join(cd.PROJECT_DIR, 'argscichat_train_dev', 'fold_{1}_{2}.json')
    config_data['train_data_path'] = base_data_path.format(experiment_params['fold'], 'train')
    config_data['validation_data_path'] = base_data_path.format(experiment_params['fold'], 'val')

    # Experiments info
    config_data['dataset_reader']['context'] = experiment_params['context']
    config_data['validation_dataset_reader']['context'] = experiment_params['context']
    config_data['pytorch_seed'] = experiment_params['pytorch_seed']
    config_data['model']['use_evidence_scaffold'] = experiment_params['use_evidence_scaffold']

    # Save config file
    training_config_path = os.path.join(base_config_path, 'current_argscichat.json')
    save_json(training_config_path, config_data)

    # Run script
    cmd = ['allennlp',
           'train',
           'training_config/current_argscichat.json',
           '-s',
           'model_runs/{}'.format(experiment_info['folder_name']),
           '--include-package',
           'argscichat_baselines']
    cmd = ' '.join(cmd)
    print('Executing command: {}'.format(cmd))
    process = Popen(cmd, cwd=cd.PROJECT_DIR, shell=True)
    process.wait()


if __name__ == '__main__':

    # Settings

    experiments = []

    model_names = [
        'query_facts'
    ]

    contexts = [
        ['query', 'facts']
    ]

    folds = [0, 1, 2, 3, 4]

    seeds = [
        15371,
        15372,
        15373
    ]

    parameters = {
        'use_evidence_scaffold': False
    }

    for model_name, model_contexts in zip(model_names, contexts):
        for seed in seeds:
            for fold in folds:
                experiments.append(
                    {
                        'folder_name': "{0}_{1}_fold_{2}".format(model_name, seed, fold),
                        'params': {
                            'fold': fold,
                            'context': model_contexts,
                            'pytorch_seed': seed,
                            **parameters
                        }
                    }
                )

    # Define manual configurations here if needed
    manual_experiments = []

    experiments += manual_experiments

    # ----
    print('*' * 50)
    print('Showing experiments')
    for experiment in experiments:
        print(experiment)
    print('*' * 50)

    input('Press any key to execute experiments...')

    for experiment_info in tqdm(experiments):
        print('Executing experiment: \n{}'.format(experiment_info))
        run_experiment(experiment_info)
        print('*' * 50)
