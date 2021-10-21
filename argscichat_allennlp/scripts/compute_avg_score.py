"""

Computes average metric score for multiple seed runs.

"""

from argscichat_allennlp.utility.json_utils import load_json
import numpy as np
import os
import argscichat_allennlp.const_define as cd

if __name__ == '__main__':

    models_path = os.path.join(cd.PROJECT_DIR, 'model_runs')
    model_name = 'query_facts_history'
    seeds = [
        15371,
        15372,
        15373
    ]
    metric_name = 'answer_f1'

    metric_values = []

    for folder in os.listdir(models_path):
        if any([folder.startswith('{0}_{1}'.format(model_name, seed)) for seed in seeds]):
            sub_path = os.path.join(models_path, folder)
            metrics_path = os.path.join(sub_path, 'metrics.json')
            metrics_data = load_json(metrics_path)

            metric_values.append(metrics_data['best_validation_{}'.format(metric_name)])

    assert len(metric_values) == len(seeds)
    print('Average ({0} models) {1} score: {2}'.format(len(metric_values), metric_name, np.mean(metric_values)))