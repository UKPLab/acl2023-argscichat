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


def run_experiment(experiment_info, sentence_split=False):

    if sentence_split:
        prefix = 'sentence_'
    else:
        prefix = ''

    # Read config file
    base_config_path = os.path.join(cd.PROJECT_DIR, 'training_config')
    config_path = os.path.join(base_config_path, 'led_{}argscichat.jsonnet'.format(prefix))
    config_data = sj.loads(_jsonnet.evaluate_file(config_path))

    # Update config file
    experiment_params = experiment_info['params']
    config_data['dataset_reader']['context'] = experiment_params['context']
    config_data['dataset_reader']['include_argument_mask'] = experiment_params['include_argument_mask']
    config_data['validation_dataset_reader']['context'] = experiment_params['context']
    config_data['validation_dataset_reader']['include_argument_mask'] = experiment_params['include_argument_mask']
    config_data['pytorch_seed'] = experiment_params['pytorch_seed']
    config_data['model']['use_evidence_scaffold'] = experiment_params['use_evidence_scaffold']
    config_data['model']['include_argument_mask'] = experiment_params['include_argument_mask']

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

    sentence_split = True

    experiments = [
        {
            'folder_name': "test_name",
            'params': {
                'context': ["query"],
                'pytorch_seed': 15371,
                'use_evidence_scaffold': False,
                'include_argument_mask': False,
                "argument_mask_threshold": "0.7"
            }
        },
    ]

    # ----

    for experiment_info in tqdm(experiments):
        print('Executing experiment: \n{}'.format(experiment_info))
        run_experiment(experiment_info, sentence_split)
        print('*' * 50)


