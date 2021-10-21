"""

Computes facts selection baselines as in QASPER dataset

Instead of paragraphs (since we don't have latex source codes) we use sentences as the basic unit

Argumentative annotations are used to filter out sentences.
Baselines can only select argumentative sentences as potential candidates

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


def random_baseline(doc_sentences, n_facts, arg_mask_indexes):
    return np.random.choice(arg_mask_indexes, size=n_facts, replace=False).tolist()


def first_sentence_baseline(doc_sentences, n_facts, arg_mask_indexes):
    return arg_mask_indexes[:n_facts]


def tfidf_baseline(doc_sentences, query, n_facts, arg_mask_indexes):
    paper_vectorizer = TfidfVectorizer(decode_error='replace',
                                       strip_accents='unicode',
                                       analyzer='word',
                                       stop_words='english')
    index = paper_vectorizer.fit_transform(doc_sentences)
    query_vector = paper_vectorizer.transform([query])
    similarities = cosine_similarity(index, query_vector).flatten()
    sorted_indexes = np.argsort(similarities, axis=0)
    sorted_indexes = [idx for idx in sorted_indexes if idx in arg_mask_indexes]
    most_similar_indexes = sorted_indexes[-n_facts:]
    return most_similar_indexes


def graph_tfidf_baseline(doc_sentences, query, n_facts, arg_mask_indexes, arg_graph, past_facts):
    past_facts = extract_past_facts(past_facts)
    using_graph = False
    neighbours = []
    neighbour_indexes = []
    if len(past_facts):
        neighbours = []
        for fact in past_facts:
            if fact not in arg_graph:
                found_keys = [key for key in arg_graph if fact in key or key in fact]
                if len(found_keys):
                    fact = found_keys[0]

            if fact in arg_graph:
                neighbours += [fact]
                neighbours += arg_graph[fact]

        if len(neighbours):
            neighbours = list(set(neighbours))
            neighbours = [sent_tokenize(sent) for sent in neighbours]
            neighbours = [sent for seq in neighbours for sent in seq if len(punctuation_filtering(sent.strip()))]

            arg_neighbours = []
            for neigh in neighbours:
                neigh = punctuation_filtering(neigh)
                for sent_idx, sent in enumerate(doc_sentences):
                    sent = punctuation_filtering(sent)
                    if neigh in sent or sent in neigh:
                        if sent_idx in arg_mask_indexes:
                            arg_neighbours.append(sent)
                            neighbour_indexes.append(sent_idx)
                            break

            if len(arg_neighbours):
                using_graph = True
                neighbours = arg_neighbours

    if not len(neighbours):
        for sent_idx, sent in enumerate(doc_sentences):
            if sent_idx in arg_mask_indexes:
                neighbours.append(sent)
                neighbour_indexes.append(sent_idx)

    paper_vectorizer = TfidfVectorizer(decode_error='replace',
                                       strip_accents='unicode',
                                       analyzer='word',
                                       stop_words='english')
    encoded_doc_sentences = paper_vectorizer.fit_transform(neighbours)
    encoded_query = paper_vectorizer.transform([query])
    similarities = cosine_similarity(encoded_doc_sentences, encoded_query).flatten()
    sorted_indexes = np.argsort(similarities, axis=0)
    sorted_indexes = [neighbour_indexes[idx] for idx in sorted_indexes]  # remapping indexes to document
    most_similar_indexes = sorted_indexes[-n_facts:]

    if len(most_similar_indexes) < n_facts:
        most_similar_indexes += [most_similar_indexes[-1]] * (n_facts - len(most_similar_indexes))

    return most_similar_indexes, using_graph


def sbert_baseline(bert_model, doc_sentences, query, n_facts, arg_mask_indexes):
    enc_doc_sentences = np.array(
        [bert_model.encode(sent, convert_to_tensor=True).cpu().numpy().ravel() for sent in doc_sentences])
    enc_query = bert_model.encode(query, convert_to_tensor=True).cpu().numpy().reshape(1, -1)
    similarities = cosine_similarity(enc_doc_sentences, enc_query).flatten()
    sorted_indexes = np.argsort(similarities, axis=0)
    sorted_indexes = [idx for idx in sorted_indexes if idx in arg_mask_indexes]
    most_similar_indexes = sorted_indexes[-n_facts:]
    return most_similar_indexes


def extract_past_facts(past_facts):
    past_facts = literal_eval(past_facts)
    if len(past_facts):
        past_facts = [item for seq in past_facts for item in seq]

    return past_facts


def graph_sbert_baseline(bert_model, doc_sentences, query, n_facts, arg_mask_indexes, arg_graph, past_facts):
    past_facts = extract_past_facts(past_facts)
    using_graph = False
    neighbours = []
    neighbour_indexes = []
    if len(past_facts):
        neighbours = []
        for fact in past_facts:
            if fact not in arg_graph:
                found_keys = [key for key in arg_graph if fact in key or key in fact]
                if len(found_keys):
                    fact = found_keys[0]

            if fact in arg_graph:
                neighbours += [fact]
                neighbours += arg_graph[fact]

        if len(neighbours):
            neighbours = list(set(neighbours))
            neighbours = [sent_tokenize(sent) for sent in neighbours]
            neighbours = [sent for seq in neighbours for sent in seq if len(punctuation_filtering(sent.strip()))]

            arg_neighbours = []
            for neigh in neighbours:
                neigh = punctuation_filtering(neigh)
                for sent_idx, sent in enumerate(doc_sentences):
                    sent = punctuation_filtering(sent)
                    if neigh in sent or sent in neigh:
                        if sent_idx in arg_mask_indexes:
                            arg_neighbours.append(sent)
                            neighbour_indexes.append(sent_idx)
                            break

            if len(arg_neighbours):
                using_graph = True
                neighbours = arg_neighbours

    if not len(neighbours):
        for sent_idx, sent in enumerate(doc_sentences):
            if sent_idx in arg_mask_indexes:
                neighbours.append(sent)
                neighbour_indexes.append(sent_idx)

    enc_doc_sentences = np.array(
        [bert_model.encode(sent, convert_to_tensor=True).cpu().numpy().ravel() for sent in neighbours])
    enc_query = bert_model.encode(query, convert_to_tensor=True).cpu().numpy().reshape(1, -1)
    similarities = cosine_similarity(enc_doc_sentences, enc_query).flatten()
    sorted_indexes = np.argsort(similarities, axis=0)
    sorted_indexes = [neighbour_indexes[idx] for idx in sorted_indexes]  # remapping indexes to document
    most_similar_indexes = sorted_indexes[-n_facts:]

    if len(most_similar_indexes) < n_facts:
        most_similar_indexes += [most_similar_indexes[-1]] * (n_facts - len(most_similar_indexes))

    return most_similar_indexes, using_graph


def build_argumentative_graph(doc_components_df):
    graph_dict = {}
    for row_idx, row in doc_components_df.iterrows():
        source = row['Source']
        target = row['Target']
        relation = row['relation_type']
        if relation == 'link':
            graph_dict.setdefault(source, []).append(target)
            graph_dict.setdefault(target, []).append(source)

    return graph_dict


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

    doc_ids = [0, 1, 2, 3, 137, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 21, 150, 23, 24, 25, 26, 149,
               22, 39, 40, 41, 42, 43, 44, 52, 180, 54, 55, 56, 181, 182, 183, 184, 75, 76, 77, 78, 79, 80, 222,
               224, 225]

    # Settings
    df_path = os.path.join(cd.LOCAL_DATASETS_DIR, 'arg_sci_chat', 'argscichat_corpus.csv')
    df = pd.read_csv(df_path)

    threshold = 0.9
    components_path = os.path.join(cd.LOCAL_DATASETS_DIR, 'arg_sci_chat',
                                   'papers_model_components_dataset_{}.csv'.format(threshold))
    components_df = pd.read_csv(components_path)

    model_name = "all-mpnet-base-v2"
    bert_model = SentenceTransformer(model_name)

    metric = MaskF1(name='mask_f1')
    # -----

    graph_usage_count = {
        'graph_sbert': 0,
        'graph_tfidf': 0
    }
    debug_graph_usage = {
        'graph_sbert': [],
        'graph_tfidf': []
    }
    avg_total_f1 = {}
    for doc_id in tqdm(doc_ids):
        doc_data = df.iloc[doc_id]
        true_facts = literal_eval(doc_data['Supporting Facts'])
        doc_title = doc_data['Article Title']
        doc_text = doc_data['Article Content']
        query = doc_data['P_Message']
        doc_sentences = sent_tokenize(doc_text)
        doc_sentences = [sent for sent in doc_sentences if len(punctuation_filtering(sent.strip()))]

        # Filter doc sentences by selecting argumentative ones only
        doc_components_df = components_df[components_df['Doc'] == doc_title]
        all_doc_arguments = doc_components_df['Source'].values.tolist() + doc_components_df['Target'].values.tolist()
        all_doc_arguments = list(set(all_doc_arguments))
        all_doc_arg_sentences = [sent_tokenize(arg) for arg in all_doc_arguments]
        all_doc_arg_sentences = [sent for seq in all_doc_arg_sentences for sent in seq if
                                 len(punctuation_filtering(sent.strip()))]

        arg_mask_indexes = [sent_idx for sent_idx, sent in enumerate(doc_sentences) if sent in all_doc_arg_sentences]

        # Remove facts that are in title
        true_facts = [sent_tokenize(fact) for fact in true_facts]
        true_facts = [sent for seq in true_facts for sent in seq if len(punctuation_filtering(sent.strip()))]
        true_facts = [x.replace("\n", " ").strip() for x in true_facts]
        true_facts = [x for x in true_facts if x != ""]
        true_facts = validate_facts(true_facts, doc_sentences, doc_title)

        n_facts = len(true_facts)

        true_indexes = retrieve_indexes(doc_sentences=doc_sentences, true_facts=true_facts)

        random_indexes = random_baseline(doc_sentences, n_facts, arg_mask_indexes)
        first_indexes = first_sentence_baseline(doc_sentences, n_facts, arg_mask_indexes)
        tfidf_indexes = tfidf_baseline(doc_sentences, query, n_facts, arg_mask_indexes)
        sbert_indexes = sbert_baseline(bert_model, doc_sentences, query, n_facts, arg_mask_indexes)

        arg_graph = build_argumentative_graph(doc_components_df)

        graph_sbert_indexes, sbert_using_graph = graph_sbert_baseline(bert_model, doc_sentences, query, n_facts,
                                                                      arg_mask_indexes, arg_graph,
                                                                      doc_data['Past Supporting Facts'])
        graph_usage_count['graph_sbert'] += int(sbert_using_graph)
        if sbert_using_graph:
            debug_graph_usage.setdefault('graph_sbert', []).append(doc_id)

        graph_tfidf_indexes, tfidf_using_graph = graph_tfidf_baseline(doc_sentences, query, n_facts, arg_mask_indexes,
                                                                      arg_graph,
                                                                      doc_data['Past Supporting Facts'])
        graph_usage_count['graph_tfidf'] += int(tfidf_using_graph)
        if tfidf_using_graph:
            debug_graph_usage.setdefault('graph_tfidf', []).append(doc_id)

        assert len(true_facts) == len(random_indexes) == len(first_indexes) == len(tfidf_indexes) == len(
            sbert_indexes) == len(graph_sbert_indexes) == len(graph_tfidf_indexes)

        avg_doc_f1 = {}
        for true_index in true_indexes:

            true_mask = [0] * len(doc_sentences)
            true_mask[true_index] = 1

            for baseline_indexes, baseline_name in zip(
                    [random_indexes, first_indexes, tfidf_indexes, sbert_indexes, graph_sbert_indexes,
                     graph_tfidf_indexes],
                    ['Random', 'First', 'TF-IDF', 'SBERT', 'Graph SBERT', 'Graph Tf-Idf']):
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

    for key, value in graph_usage_count.items():
        print('Graph usage documents: {}'.format(debug_graph_usage[key]))
        print(value, len(doc_ids))
        print('Graph usage ratio: {0}'.format(value / len(doc_ids)))
