"""

Evaluation script for computing model sperformance when trained on sentence-split ArgSciChat.

"""

import os
from typing import List

import numpy as np
from allennlp_models.rc.tools import squad
from tqdm import tqdm

import argscichat_allennlp.const_define as cd
from argscichat_allennlp.utility.json_utils import load_json, load_jsonl


def _compute_evidence_f1(
        predicted_evidence_indices: List[List[int]],
        gold_evidence_indices: List[List[List[int]]]
) -> List[float]:
    f1s = []
    for instance_predicted, instance_gold in zip(predicted_evidence_indices, gold_evidence_indices):
        instance_f1s = []
        for gold in instance_gold:
            # Skip example without facts
            if sum(gold) == 0:
                break
            # If the document was truncated to fit in the model, the gold will be longer than the
            # predicted indices.
            predicted = instance_predicted + [0] * (len(gold) - len(instance_predicted))
            true_positives = sum([i and j for i, j in zip(predicted, gold)])
            if sum(predicted) == 0:
                precision = 1.0 if sum(gold) == 0 else 0.0
            else:
                precision = true_positives / sum(predicted)
            recall = true_positives / sum(gold) if sum(gold) != 0 else 1.0
            if precision + recall == 0:
                instance_f1s.append(0.0)
            else:
                instance_f1s.append(2 * precision * recall / (precision + recall))
        if len(instance_f1s):
            f1s.append(max(instance_f1s))
    return f1s


def aggregate_and_evaluate_predictions(filepath, ground_truth_dict, best_solution=False):
    predictions = load_jsonl(filepath)
    avg_f1 = []
    avg_facts_f1 = []

    for pair_id, pair_truth in ground_truth_dict.items():
        found_predictions = []
        for pred in predictions:
            base_pred_id = pred['pair_id'].split('_')
            base_pred_id = '_'.join(base_pred_id[:-1])
            if pair_id == base_pred_id:
                found_predictions.append(pred)

        pred_facts_mask = None

        if len(found_predictions) > 1:
            found_predictions = sorted(found_predictions, key=lambda item: int(item['pair_id'].split('_')[-1]))
            unified_prediction = ' '.join(pred['predicted_answer'] for pred in found_predictions)

            if 'predicted_evidence' in found_predictions[0]:
                if not best_solution:
                    pred_facts_mask = np.zeros((len(pair_truth['facts_mask'])))
                    for pred in found_predictions:
                        # In case of history, we need to truncate first mask indices
                        predicted_facts = pred['predicted_evidence']
                        difference = len(predicted_facts) - len(pair_truth['facts_mask'])
                        assert difference >= 0
                        if difference > 0:
                            predicted_facts = predicted_facts[difference:]

                        pred_facts_mask += np.array(predicted_facts)
                    pred_facts_mask[pred_facts_mask > 1] = 1
                    pred_facts_mask = pred_facts_mask.tolist()
                else:
                    pred_facts_mask = []
                    for pred in found_predictions:
                        # In case of history, we need to truncate first mask indices
                        predicted_facts = pred['predicted_evidence']
                        difference = len(predicted_facts) - len(pair_truth['facts_mask'])
                        assert difference >= 0
                        if difference > 0:
                            predicted_facts = predicted_facts[difference:]

                        pred_facts_mask.append(predicted_facts)
        else:
            unified_prediction = found_predictions[0]['predicted_answer']

            if 'predicted_evidence' in found_predictions[0]:
                pred_facts_mask = np.array(found_predictions[0]['predicted_evidence']).tolist()

        f1 = squad.compute_f1(unified_prediction, pair_truth['reply'])
        avg_f1.append(f1)

        if pred_facts_mask is not None:
            if type(pred_facts_mask[0]) == list:
                best_fact_f1 = None
                for pred_mask in pred_facts_mask:
                    facts_f1 = _compute_evidence_f1([pred_mask], [[pair_truth['facts_mask']]])
                    if best_fact_f1 is None or (best_fact_f1 is not None and len(facts_f1) and best_fact_f1[0] < facts_f1[0]):
                        best_fact_f1 = facts_f1

                if len(best_fact_f1):
                    avg_facts_f1.append(best_fact_f1[0])
            else:
                facts_f1 = _compute_evidence_f1([pred_facts_mask], [[pair_truth['facts_mask']]])
                if len(facts_f1):
                    avg_facts_f1.append(facts_f1[0])
        else:
            avg_facts_f1.append(0.0)

    return np.mean(avg_f1), np.mean(avg_facts_f1)


def retrieve_ground_truth(validation_set):
    ground_truth_dict = {}
    for article_title, article_data in validation_set.items():
        pairs = article_data['message_pairs']
        for pair in pairs:
            reply = pair['DE_Message']
            facts = pair['facts']
            facts_mask = [0] * len(article_data['content'])
            for fact in facts:
                for sent_idx, sent in enumerate(article_data['content']):
                    if sent in fact or fact in sent:
                        facts_mask[sent_idx] = 1
                        break

            ground_truth_dict[pair['id']] = {
                'reply': reply,
                'facts_mask': facts_mask
            }

    return ground_truth_dict


if __name__ == '__main__':

    # Settings

    predictions_path = os.path.join(cd.PROJECT_DIR, 'model_predictions')

    validation_set = load_json(os.path.join(cd.PROJECT_DIR, 'argscichat_train_dev', 'val.json'))
    ground_truth_dict = retrieve_ground_truth(validation_set)

    seeds = [
        "15371",
        "15372",
        "15373"
    ]

    test_names = [
    ]

    is_pipeline = False

    if is_pipeline:
        prefix = 'pipeline_'
    else:
        prefix = ''

    # Enable it to compute fact-f1 on the best split fact selection predictions
    # (instead of merging splits predictions together)
    best_solution = False

    # ----

    reply_performance_dict = {}
    facts_selection_performance_dict = {}
    for name in tqdm(test_names):
        for seed in seeds:
            folder = '{0}_{1}'.format(name, seed)
            sub_path = os.path.join(predictions_path, folder, '{0}predictions.jsonl'.format(prefix))
            model_f1, model_facts_f1 = aggregate_and_evaluate_predictions(sub_path, ground_truth_dict, best_solution)

            for seed in seeds:
                if seed in folder:
                    model_group = folder.split('_{}'.format(seed))[0]
                    reply_performance_dict.setdefault(model_group, []).append(model_f1)
                    facts_selection_performance_dict.setdefault(model_group, []).append(model_facts_f1)

    reply_performance_dict = sorted(reply_performance_dict.items(), key=lambda x: np.mean(x[1]))
    facts_selection_performance_dict = sorted(facts_selection_performance_dict.items(), key=lambda x: np.mean(x[1]))

    for key, value in reply_performance_dict:
        print('Model {0} ({2} runs):'
              ' Reply F1 = {1}'.format(key,
                                       np.mean(value),
                                       len(value)))

    print()
    print('*' * 50)
    print()

    for key, value in facts_selection_performance_dict:
        print('Model {0} ({2} runs):'
              ' Facts Selection F1 = {1}'.format(key,
                                                 np.mean(value),
                                                 len(value)))
