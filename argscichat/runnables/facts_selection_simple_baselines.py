"""

Computes facts selection baselines as in QASPER dataset

Instead of paragraphs (since we don't have latex source codes) we use sentences as the basic unit

"""

import os
from ast import literal_eval

import numpy as np
import pandas as pd
from nltk import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

import argscichat.const_define as cd
from argscichat.generic.metrics import MaskF1
from argscichat.utility.preprocessing_utils import punctuation_filtering


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
    for sent_idx, sent in enumerate(doc_sentences):
        for true_fact in true_facts:
            if sent in true_fact or true_fact in sent:
                true_indexes.append(sent_idx)

    assert len(true_facts) == len(true_indexes)
    return true_indexes


def validate_facts(facts, doc_sentences, title):
    # Validate facts
    missing_indexes = []
    for fact_idx, fact in enumerate(facts):
        found = False
        for paragraph in doc_sentences:
            if fact.lower() in paragraph.lower() or paragraph.lower() in fact.lower():
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
            print('Inconsistency remains ({0} candidates): \n{1}'.format(len(missing_indexes) - len(to_avoid),
                                                                         missing_facts))

    facts_spans = [facts[idx] for idx in range(len(facts)) if idx not in missing_indexes]
    return facts_spans


if __name__ == '__main__':

    # Settings
    doc_ids = [0, 1, 2, 3, 137, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 21, 150, 23, 24, 25, 26, 149,
               22, 39, 40, 41, 42, 43, 44, 52, 180, 54, 55, 56, 181, 182, 183, 184, 75, 76, 77, 78, 79, 80, 222,
               224, 225]

    df_path = os.path.join(cd.LOCAL_DATASETS_DIR, 'arg_sci_chat', 'argscichat_corpus.csv')
    df = pd.read_csv(df_path)

    metric = MaskF1(name='mask_f1')

    model_name = "all-mpnet-base-v2"
    bert_model = SentenceTransformer(model_name)
    # -----

    avg_total_f1 = {}
    for doc_id in tqdm(doc_ids):
        doc_data = df.iloc[doc_id]
        true_facts = literal_eval(doc_data['Supporting Facts'])
        doc_title = doc_data['Article Title']
        doc_text = doc_data['Article Content']
        query = doc_data['P_Message']
        doc_sentences = sent_tokenize(doc_text)
        doc_sentences = [sent for sent in doc_sentences if len(punctuation_filtering(sent.strip()))]

        # Remove facts that are in title
        true_facts = [sent_tokenize(fact) for fact in true_facts]
        true_facts = [sent for seq in true_facts for sent in seq if len(punctuation_filtering(sent.strip()))]
        true_facts = [x.replace("\n", " ").strip() for x in true_facts]
        true_facts = [x for x in true_facts if x != ""]
        true_facts = validate_facts(true_facts, doc_sentences, doc_title)

        n_facts = len(true_facts)

        true_indexes = retrieve_indexes(doc_sentences=doc_sentences, true_facts=true_facts)

        random_indexes = random_baseline(doc_sentences, n_facts)
        first_indexes = first_sentence_baseline(doc_sentences, n_facts)
        tfidf_indexes = tfidf_baseline(doc_sentences, query, n_facts)
        sbert_indexes = sbert_baseline(bert_model, doc_sentences, query, n_facts)

        assert len(true_facts) == len(random_indexes) == len(first_indexes) == len(tfidf_indexes) == len(sbert_indexes)

        avg_doc_f1 = {}
        for true_index in true_indexes:

            true_mask = [0] * len(doc_sentences)
            true_mask[true_index] = 1

            for baseline_indexes, baseline_name in zip([random_indexes, first_indexes, tfidf_indexes, sbert_indexes],
                                                       ['Random', 'First', 'TF-IDF', 'SBERT']):
                best_f1 = 0.0
                for pred_index in baseline_indexes:
                    pred_mask = [0] * len(doc_sentences)
                    pred_mask[pred_index] = 1
                    fact_f1 = metric(y_pred=pred_mask, y_true=true_mask)
                    if fact_f1 > best_f1:
                        best_f1 = fact_f1

                avg_doc_f1.setdefault(baseline_name, []).append(best_f1)

        for baseline_name in avg_doc_f1.keys():
            avg_total_f1.setdefault(baseline_name, []).append(np.mean(avg_doc_f1[baseline_name]))

    for baseline_name in avg_total_f1.keys():
        print('{0} --> {1}'.format(baseline_name, np.mean(avg_total_f1[baseline_name])))
