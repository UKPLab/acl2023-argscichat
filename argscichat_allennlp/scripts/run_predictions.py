"""

Simple utility script for running multiple predictions in a sequence

"""

import os
from subprocess import Popen

from tqdm import tqdm

import argscichat_allennlp.const_define as cd


def run_prediction(folder):
    os.makedirs(os.path.join(cd.PROJECT_DIR, 'model_predictions', folder))

    # Run script
    cmd = ['allennlp',
           'predict',
           '--output-file',
           'model_predictions/{}/predictions.json'.format(folder),
           '--predictor',
           'argscichat',
           '--use-dataset-reader',
           '--silent',
           '--include-package',
           'argscichat_baselines',
           'model_runs/{}'.format(folder),
           os.path.join('argscichat_train_dev', 'sentence_val.json')
           ]
    cmd = ' '.join(cmd)
    print('Executing command: {}'.format(cmd))
    process = Popen(cmd, cwd=cd.PROJECT_DIR, shell=True)
    process.wait()


if __name__ == '__main__':

    models_path = os.path.join(cd.PROJECT_DIR, 'model_runs')
    force_prediction = False

    for folder in tqdm(os.listdir(models_path)):
        if force_prediction or not os.path.isfile(
                os.path.join(cd.PROJECT_DIR, 'model_predictions', folder, 'predictions.json')):
            run_prediction(folder)
