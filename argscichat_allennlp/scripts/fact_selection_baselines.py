import os
import string

import numpy as np
from nltk import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity

import argscichat_allennlp.const_define as cd
from argscichat_allennlp.utility.json_utils import load_json
from tqdm import tqdm


class Metric(object):

    def __init__(self, name):
        self.name = name

    def __call__(self, y_pred, y_true, **kwargs):
        raise NotImplementedError()

    def retrieve_parameters_from_network(self, network):
        pass

    def reset(self):
        raise NotImplementedError()


class MaskF1(Metric):

    def __init__(self, average='binary', **kwargs):
        super(MaskF1, self).__init__(**kwargs)
        self.average = average

    def __call__(self, y_pred, y_true, **kwargs):
        y_pred = np.array(y_pred) if type(y_pred) != np.ndarray else y_pred
        y_true = np.array(y_true) if type(y_true) != np.ndarray else y_true

        y_pred = y_pred.ravel()
        y_true = y_true.ravel()

        return f1_score(y_true=y_true, y_pred=y_pred, average=self.average, **kwargs)


def random_baseline(doc_sentences, n_facts):
    return np.random.choice(np.arange(len(doc_sentences)), size=n_facts, replace=False).tolist()


def first_sentence_baseline(doc_sentences, n_facts):
    return np.arange(n_facts)


def tfidf_baseline(doc_sentences, query, n_facts):
    paper_vectorizer = TfidfVectorizer(decode_error='replace',
                                       strip_accents='unicode',
                                       analyzer='word',
                                       stop_words='english')
    index = paper_vectorizer.fit_transform(doc_sentences)
    query_vector = paper_vectorizer.transform([query])
    similarities = cosine_similarity(index, query_vector).flatten()
    most_similar_indexes = np.argsort(similarities, axis=0)[-n_facts:]
    return most_similar_indexes


def sbert_baseline(bert_model, doc_sentences, query, n_facts):
    enc_doc_sentences = np.array(
        [bert_model.encode(sent, convert_to_tensor=True).cpu().numpy().ravel() for sent in doc_sentences])
    enc_query = bert_model.encode(query, convert_to_tensor=True).cpu().numpy().reshape(1, -1)
    similarities = cosine_similarity(enc_doc_sentences, enc_query).flatten()
    most_similar_indexes = np.argsort(similarities, axis=0)[-n_facts:]
    return most_similar_indexes


def retrieve_indexes(doc_sentences, true_facts):
    true_indexes = []
    for true_fact in true_facts:
        filtered_true_fact = punctuation_filtering(true_fact.lower().strip())
        for sent_idx, sent in enumerate(doc_sentences):
            filtered_sent = punctuation_filtering(sent.lower().strip())
            if filtered_true_fact in filtered_sent:
                true_indexes.append(sent_idx)
                break

    assert len(true_facts) == len(true_indexes), f"Got: Facts={len(true_facts)} -- Indexes={len(true_indexes)}"
    return true_indexes


def validate_facts(facts, doc_sentences, title):
    # Validate facts
    missing_indexes = []
    for fact_idx, fact in enumerate(facts):
        found = False
        filtered_fact = punctuation_filtering(fact.lower().strip())
        for paragraph in doc_sentences:
            filtered_paragraph = punctuation_filtering(paragraph.lower().strip())
            if filtered_fact in filtered_paragraph:
                found = True
                break

        if not found:
            missing_indexes.append(fact_idx)

    if len(missing_indexes) > 0:
        print('Found potential issue concerning facts retrieval: Got {0} -- Expected {1}.'
              ' Verifying if missing fact(s) are in title...'.format(len(facts) - len(missing_indexes),
                                                                     len(facts)))
        to_avoid = []
        for fact_idx in missing_indexes:
            if facts[fact_idx] in title:
                to_avoid.append(fact_idx)

        if len(to_avoid) == len(missing_indexes):
            print('Inconsistency has been debunked!')
        else:
            missing_facts = [facts[idx] for idx in missing_indexes if idx not in to_avoid]
            raise RuntimeError('Inconsistency remains ({0} candidates): \n{1}'.format(len(missing_indexes) - len(to_avoid),
                                                                         missing_facts))

    facts_spans = [facts[idx] for idx in range(len(facts)) if idx not in missing_indexes]
    return facts_spans


def punctuation_filtering(line):
    """
    Filters given sentences by removing punctuation
    """

    table = str.maketrans('', '', string.punctuation)
    trans = [w.translate(table) for w in line.split()]

    return ''.join([w for w in trans if w != ''])


if __name__ == '__main__':

    # Setup
    dataset_path = os.path.join(cd.PROJECT_DIR, 'argscichat_train_dev')

    val_metric_values = []
    test_metric_values = []

    # On which data split we want to run baselines
    data_split = 'test'

    folds = [
        0,
        1,
        2,
        3,
        4
    ]

    # Evaluation
    metric = MaskF1(name='mask_f1')

    # S-BERT
    model_name = "all-mpnet-base-v2"
    bert_model = SentenceTransformer(model_name)

    # facts for retrieval
    n_facts = 2

    avg_fold_f1 = {}
    for fold in tqdm(folds):
        # Load test data
        test_data_path = os.path.join(dataset_path, 'fold_{0}_{1}.json'.format(fold, data_split))
        test_data = load_json(test_data_path)

        avg_fact_f1 = {}
        for paper_title, paper_data in test_data.items():
            print('Considering paper: ', paper_title)

            paper_sentences = paper_data['content']

            for pair in paper_data['message_pairs']:
                query = pair['P_Message']
                true_facts = pair['facts']

                # Skip messages without facts
                if not len(true_facts):
                    continue

                # Remove facts that are in title
                true_facts = [sent_tokenize(fact) for fact in true_facts]
                true_facts = [sent for seq in true_facts for sent in seq if len(punctuation_filtering(sent.strip()))]
                true_facts = [x.replace("\n", " ").strip() for x in true_facts]
                true_facts = [x for x in true_facts if x != ""]
                true_facts = validate_facts(true_facts, paper_sentences, paper_title)

                n_true_facts = len(true_facts)

                if n_true_facts == 0:
                    continue

                true_indexes = retrieve_indexes(doc_sentences=paper_sentences, true_facts=true_facts)

                # Run baselines
                random_indexes = random_baseline(paper_sentences, n_facts)
                first_indexes = first_sentence_baseline(paper_sentences, n_facts)
                tfidf_indexes = tfidf_baseline(paper_sentences, query, n_facts)
                sbert_indexes = sbert_baseline(bert_model, paper_sentences, query, n_facts)
                assert len(random_indexes) == len(first_indexes) == len(tfidf_indexes) == len(
                    sbert_indexes), f"Got: Random={len(random_indexes)}" \
                                    f" -- First={len(first_indexes)} -- Tf-Idf={len(tfidf_indexes)}" \
                                    f" -- S-BERT={len(sbert_indexes)}. N_fact={n_facts}, Paper Sentences={len(paper_sentences)}"

                true_mask = np.array([0] * len(paper_sentences))
                true_mask[true_indexes] = 1

                for baseline_indexes, baseline_name in zip(
                        [random_indexes, first_indexes, tfidf_indexes, sbert_indexes],
                        ['Random', 'First', 'TF-IDF', 'SBERT']):
                    pred_mask = [1 if idx in baseline_indexes else 0 for idx in range(len(paper_sentences))]
                    fact_f1 = metric(y_pred=pred_mask, y_true=true_mask)

                    avg_fact_f1.setdefault(baseline_name, []).append(fact_f1)

        for baseline_name in avg_fact_f1.keys():
            avg_fold_f1.setdefault(baseline_name, []).append(np.mean(avg_fact_f1[baseline_name]))

    for baseline_name in avg_fold_f1.keys():
        print('{0} --> {1}'.format(baseline_name, np.mean(avg_fold_f1[baseline_name])))
