"""

Simple utility script for running multiple predictions in a sequence

"""

import os
from subprocess import Popen

from tqdm import tqdm

import argscichat_allennlp.const_define as cd


def run_prediction(models_path, folder, sentence_split):
    base_path = os.path.join(cd.PROJECT_DIR, 'model_predictions', folder)
    if not os.path.isdir(base_path):
        os.makedirs(base_path)

    if sentence_split:
        prefix = 'sentence_'
    else:
        prefix = ''

    # Run script
    cmd = ['allennlp',
           'predict',
           '--output-file',
           'model_predictions/{}/predictions.jsonl'.format(folder),
           '--predictor',
           'argscichat',
           '--use-dataset-reader',
           '--silent',
           '--include-package',
           'argscichat_baselines',
           os.path.join(models_path, folder),
           os.path.join('argscichat_train_dev', '{}val.json'.format(prefix))
           ]
    cmd = ' '.join(cmd)
    print('Executing command: {}'.format(cmd))
    process = Popen(cmd, cwd=cd.PROJECT_DIR, shell=True)
    process.wait()


if __name__ == '__main__':

    # Settings
    base_predictions_folder = os.path.join(cd.PROJECT_DIR, 'model_predictions')
    models_path = os.path.join(cd.PROJECT_DIR, 'model_runs')

    test_names = [
    ]

    seeds = [
        15371,
        15372,
        15373
    ]

    force_prediction = True
    sentence_split = True

    for name in tqdm(test_names):
        for seed in seeds:
            folder = '{0}_{1}'.format(name, seed)
            if force_prediction or not os.path.isfile(
                    os.path.join(base_predictions_folder, folder, 'predictions.json')):
                run_prediction(models_path, folder, sentence_split)
