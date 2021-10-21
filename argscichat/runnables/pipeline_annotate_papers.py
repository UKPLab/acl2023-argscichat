"""

For each paper, builds and plots a graph where nodes are sentences and links are determined
using the relation_type label

Note: The script is definitely inefficient, it might take some minutes. I would suggest to save collected statistics
to avoid re-running the script multiple times and save time.

"""

import os
from collections import Counter
from itertools import combinations
from subprocess import Popen

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import tensorflow as tf
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tqdm import tqdm

import argscichat.const_define as cd
from argscichat.generic.data_loader import DataLoaderFactory, DrInventorComponentsLoader
from argscichat.generic.processor import ProcessorFactory
from argscichat.tf_scripts.converter import TFConverterFactory
from argscichat.tf_scripts.models import TFModelFactory
from argscichat.tf_scripts.tokenizer import TFTokenizerFactory
from argscichat.utility.json_utils import load_json
from argscichat.utility.json_utils import save_json
from argscichat.utility.python_utils import flatten, merge
from argscichat.utility.tensorflow_utils import get_dataset_fn

plt.style.use('seaborn-paper')

SMALL_SIZE = 3
MEDIUM_SIZE = 10
BIGGER_SIZE = 30

plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

components_labels = ['background_claim', 'own_claim', 'data']
bio_labels = ['B', 'I', 'O']

plot_markers = [
    'X',
    'o',
    '^',
    's',
    'P',
    'd',
    '7'
]


def limit_gpu_usage(limit_gpu_visibility=False, gpu_start_index=None, gpu_end_index=None):
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if limit_gpu_visibility:
        assert gpu_start_index is not None
        assert gpu_end_index is not None
        tf.config.set_visible_devices(gpus[gpu_start_index:gpu_end_index], "GPU")  # avoid other GPUs
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def get_data_config_id(filepath, config):
    config_path = os.path.join(filepath, cd.JSON_MODEL_DATA_CONFIGS_NAME)

    if not os.path.isdir(filepath):
        os.makedirs(filepath)

    if os.path.isfile(config_path):
        data_config = load_json(config_path)
    else:
        data_config = {}

    if config in data_config:
        return int(data_config[config])
    else:
        max_config = list(map(lambda item: int(item), data_config.values()))

        if len(max_config) > 0:
            max_config = max(max_config)
        else:
            max_config = -1

        data_config[config] = max_config + 1

        save_json(config_path, data_config)

        return max_config + 1


def determine_drinventor_graph_measures():
    drinventor_loader = DrInventorComponentsLoader()
    drinventor_data = drinventor_loader.load(distinguish_reverse_pairs=False)

    train_df = drinventor_data.train_data

    # Filter to only take into account the first paragraph (Introduction)
    train_df = train_df[
        (train_df['source_ID'].str.split('_').str[1] == '0') & (train_df['target_ID'].str.split('_').str[1] == '0')]

    argument_labels = {}
    for doc in train_df['text_ID'].values:
        doc_sources = train_df[train_df['text_ID'] == doc]['source_proposition'].values
        doc_source_labels = train_df[train_df['text_ID'] == doc]['source_type'].values
        doc_targets = train_df[train_df['text_ID'] == doc]['target_proposition'].values
        doc_target_labels = train_df[train_df['text_ID'] == doc]['target_type'].values

        doc_labels = {}
        for sent, sent_label in zip(doc_sources.tolist() + doc_targets.tolist(),
                                    doc_source_labels.tolist() + doc_target_labels.tolist()):
            doc_labels.setdefault(sent, sent_label)

        argument_labels[doc] = list(doc_labels.values())

    train_df['Source'] = train_df['source_proposition']
    train_df['Target'] = train_df['target_proposition']
    overall_arg_distribution, per_doc_arg_distribution = evaluate_arguments_distribution(argument_labels)

    train_df['Doc'] = train_df['text_ID']
    doc_links, overall_link_distribution, per_doc_link_distribution = extract_argumentative_links(train_df)

    return overall_arg_distribution, per_doc_arg_distribution, overall_link_distribution, per_doc_link_distribution


# Token-level
def load_tokens_annotation_data(model_type, test_name, data_loader_type, save_prefix=None,
                                test_prefix=None, repetition_prefix=None,
                                evaluation_folder=cd.TRAIN_AND_TEST_DIR):
    data_loader_config = load_json(os.path.join(cd.CONFIGS_DIR, cd.JSON_DATA_LOADER_CONFIG_NAME))
    data_loader_info = data_loader_config['configs'][data_loader_type]
    data_loader = DataLoaderFactory.factory(cl_type=data_loader_type)
    data_handle = data_loader.load()

    model_path = os.path.join(evaluation_folder, model_type, test_name)
    network_args = load_json(os.path.join(model_path, cd.JSON_DISTRIBUTED_MODEL_CONFIG_NAME))
    training_config = load_json(os.path.join(model_path, cd.JSON_TRAINING_CONFIG_NAME))
    distributed_info = load_json(os.path.join(model_path, cd.JSON_DISTRIBUTED_CONFIG_NAME))

    config_args = {key: arg['value']
                   for key, arg in network_args.items()
                   if 'processor' in arg['flags']
                   or 'tokenizer' in arg['flags']
                   or 'converter' in arg['flags']
                   or 'data_loader' in arg['flags']}
    config_args = flatten(config_args)
    config_args = merge(config_args, data_loader_info)
    config_args_tuple = [(key, value) for key, value in config_args.items()]
    config_args_tuple = sorted(config_args_tuple, key=lambda item: item[0])

    trained_data_loader_config = load_json(os.path.join(model_path, cd.JSON_DATA_LOADER_CONFIG_NAME))
    trained_data_loader_type = trained_data_loader_config['type']
    trained_data_loader = DataLoaderFactory.factory(cl_type=trained_data_loader_type)
    trained_data_handle = trained_data_loader.load()

    config_name = '_'.join(['{0}-{1}'.format(name, value) for name, value in config_args_tuple])
    model_base_path = os.path.join(cd.TESTS_DATA_DIR,
                                   trained_data_handle.data_name,
                                   model_type)
    config_id = get_data_config_id(filepath=model_base_path, config=config_name)
    model_test_path = os.path.join(model_base_path, str(config_id))

    # Build tokenizer
    tokenizer_type = cd.MODEL_CONFIG[model_type]['tokenizer']
    tokenizer_args = {key: arg['value'] for key, arg in network_args.items() if 'tokenizer' in arg['flags']}
    tokenizer = TFTokenizerFactory.factory(tokenizer_type, **tokenizer_args)

    processor_type = cd.MODEL_CONFIG[model_type]['processor']
    processor_args = {key: arg['value'] for key, arg in network_args.items() if 'processor' in arg['flags']}
    processor_args['loader_info'] = trained_data_handle.get_additional_info()
    processor_args['retrieve_label'] = False
    processor = ProcessorFactory.factory(processor_type, **processor_args)

    # Build converter
    converter_type = cd.MODEL_CONFIG[model_type]['converter']
    converter_args = {key: arg['value'] for key, arg in network_args.items() if 'converter' in arg['flags']}
    converter = TFConverterFactory.factory(converter_type, **converter_args)

    conversion_args = converter.load_conversion_args(filepath=model_test_path, prefix='{0}_{1}'.format(
        test_prefix,
        save_prefix)
    if test_prefix is not None
    else save_prefix)

    label_map = conversion_args['label_map']
    inv_label_map = {class_key: {value: key for key, value in class_value.items()} for class_key, class_value in
                     label_map.items()}

    save_name = model_type
    if repetition_prefix is not None:
        save_name += '_repetition_{}'.format(repetition_prefix)
    if save_prefix is not None:
        save_name += '_key_{}'.format(save_prefix)

    # Processor
    test_name = 'test_data' if save_prefix is None else 'test_data_{}'.format(save_prefix)
    test_filepath = os.path.join(model_test_path, test_name)

    tokenizer_info = TFTokenizerFactory.get_supported_values()[tokenizer_type].load_info(filepath=model_test_path,
                                                                                         prefix='{0}_{1}'.format(
                                                                                             test_prefix,
                                                                                             save_prefix)
                                                                                         if test_prefix is not None
                                                                                         else save_prefix)
    converter_info = TFConverterFactory.get_supported_values()[converter_type].load_conversion_args(
        filepath=model_test_path,
        prefix='{0}_{1}'.format(test_prefix, save_prefix) if test_prefix is not None else save_prefix)

    tokenizer.initialize_with_info(tokenizer_info)
    for key, value in converter_info.items():
        setattr(converter, key, value)

    test_data = get_dataset_fn(filepath=test_filepath,
                               batch_size=training_config['batch_size'],
                               name_to_features=converter.feature_class.get_mappings(converter_info, has_labels=False),
                               selector=converter.feature_class.get_dataset_selector(processor.get_labels()),
                               is_training=False,
                               prefetch_amount=distributed_info['prefetch_amount'])

    # Building network

    network_retrieved_args = {key: value['value'] for key, value in network_args.items()
                              if 'model_class' in value['flags']}
    network_retrieved_args['additional_data'] = data_handle.get_additional_info()
    network_retrieved_args['name'] = cd.MODEL_CONFIG[model_type]['save_suffix']
    if repetition_prefix is not None:
        network_retrieved_args['name'] += '_repetition_{}'.format(repetition_prefix)
    network = TFModelFactory.factory(cl_type=model_type, **network_retrieved_args)

    text_info = merge(tokenizer_info, converter_info)
    text_info = merge(text_info, training_config)
    network.build_model(text_info=text_info)

    # Setup model by feeding an input
    network.predict(x=iter(test_data()), steps=1)

    # load pre-trained weights
    pretrained_weight_filename = network_retrieved_args['name']
    if save_prefix:
        pretrained_weight_filename += '_key_{}'.format(save_prefix)
    current_weight_filename = os.path.join(model_path,
                                           '{}.h5'.format(pretrained_weight_filename))
    network.load(os.path.join(model_path, current_weight_filename))

    # Load dataset
    df_path = os.path.join(cd.LOCAL_DATASETS_DIR, 'arg_sci_chat', 'papers_token_dataset_chunked.csv')
    df = pd.read_csv(df_path)

    return {
        'network': network,
        'processor': processor,
        'tokenizer': tokenizer,
        'converter': converter,
        'df': df,
        'inv_label_map': inv_label_map
    }


def annotate_tokens_corpus(network, tokenizer, processor, converter, df, inv_label_map, threshold, null_label=None,
                           is_bert_model=False):
    df_copy = df.copy(deep=True)
    documents = set(df_copy['Doc'].values)

    label_key = 'Merged_Arg_Label'

    for document in tqdm(documents):
        doc_df = df_copy[df_copy['Doc'] == document]
        doc_chunks = doc_df['Chunk_Index'].values
        doc_min_chunk_idx, doc_max_chunk_idx = min(doc_chunks), max(doc_chunks)

        for chunk_idx in range(doc_min_chunk_idx, doc_max_chunk_idx + 1):
            chunk_df = doc_df[doc_df['Chunk_Index'] == chunk_idx]
            example = processor.wrap_single_example(chunk_df)
            example_tf = converter.convert_example(example=example,
                                                   tokenizer=tokenizer,
                                                   label_list=processor.get_labels(),
                                                   has_labels=False)
            predictions, model_additional_info = network.batch_predict(example_tf)
            predictions = network._parse_predictions(predictions[label_key], model_additional_info).numpy().ravel()
            raw_predictions = np.max(model_additional_info['raw_predictions'][label_key].numpy(), axis=-1).ravel()
            tokens_mask = model_additional_info['tokens_mask'].numpy().ravel()

            # Remove padding
            predictions = predictions[tokens_mask == 1]
            predictions = np.array([inv_label_map[label_key][item] for item in predictions])

            raw_predictions = raw_predictions[tokens_mask == 1]
            invalid_indexes = np.where(raw_predictions < threshold)[0]
            predictions[invalid_indexes] = null_label

            if is_bert_model:
                predictions = predictions[1:-1]
                raw_predictions = raw_predictions[1:-1]

            # Find tokenization mapping
            tokens = tokenizer.convert_tokens_to_ids(example.tokens)
            token_map = [[idx] * len(item) for idx, item in enumerate(tokens)]
            token_map = np.array([item for seq in token_map for item in seq])

            assert len(token_map) == len(predictions) == len(raw_predictions)

            merged_predictions = []
            for map_idx in range(max(token_map) + 1):
                group_indexes = np.where(token_map == map_idx)
                group_predictions = predictions[group_indexes]

                if len(group_predictions) == 1:
                    merged_predictions.append(group_predictions[0])
                else:
                    majority_label = Counter(group_predictions).most_common(1)[0][0]
                    merged_predictions.append(majority_label)

            assert len(merged_predictions) == chunk_df.shape[0]

            df_copy.loc[
                (df_copy['Doc'] == document) & (df_copy['Chunk_Index'] == chunk_idx), label_key] = merged_predictions

    return df_copy


def extract_token_level_arguments(tokens_annotated_df, min_length_perc=0.20, min_tokens_length=5,
                                  strategy='most_common'):
    arguments = {}
    unlabelled_text = {}
    argument_labels = {}
    documents = set(tokens_annotated_df['Doc'].values)

    detokenizer = TreebankWordDetokenizer()

    for document in tqdm(documents):
        doc_df = tokens_annotated_df[tokens_annotated_df['Doc'] == document]
        doc_chunks = doc_df['Chunk_Index'].values
        min_chunk_idx, max_chunk_idx = min(doc_chunks), max(doc_chunks)

        for chunk_idx in range(min_chunk_idx, max_chunk_idx + 1):
            chunk_df = doc_df[doc_df['Chunk_Index'] == chunk_idx]
            chunk_sent_indexes = chunk_df['Sent_Token_Index'].values

            for sent_idx in set(chunk_sent_indexes):
                sent_tokens = chunk_df[chunk_df['Sent_Token_Index'] == sent_idx]['Token'].values
                sent_labels = chunk_df[chunk_df['Sent_Token_Index'] == sent_idx]['Merged_Arg_Label'].values

                # Uniform labels
                for label in components_labels:
                    sent_labels[sent_labels == 'B_{}'.format(label)] = label
                    sent_labels[sent_labels == 'I_{}'.format(label)] = label

                sent_counter = Counter(sent_labels)

                if strategy == 'most_common':
                    candidate = '/'
                    for tag in components_labels:
                        if sent_counter[tag] / len(sent_tokens) >= min_length_perc and len(
                                sent_tokens) >= min_tokens_length:
                            candidate = tag

                    if candidate != '/':
                        arguments.setdefault(document, []).append(detokenizer.detokenize(sent_tokens))
                        argument_labels.setdefault(document, []).append(candidate)
                    else:
                        unlabelled_text.setdefault(document, []).append(detokenizer.detokenize(sent_tokens))
                else:
                    raise NotImplementedError()

    return arguments, argument_labels, unlabelled_text


# Component-Level


def load_components_annotation_data(model_type, test_name, threshold, save_prefix=None,
                                    data_loader_type=None,
                                    test_prefix=None, repetition_prefix=None,
                                    training_test='train_and_test'):
    training_test = cd.TEST_INFO[training_test]['folder']
    model_path = os.path.join(training_test, model_type, test_name)

    train_data_loader_config = load_json(os.path.join(model_path, cd.JSON_DATA_LOADER_CONFIG_NAME))
    train_data_loader_info = train_data_loader_config['configs'][train_data_loader_config['type']]

    network_args = load_json(os.path.join(model_path, cd.JSON_DISTRIBUTED_MODEL_CONFIG_NAME))

    config_args = {key: arg['value']
                   for key, arg in network_args.items()
                   if 'processor' in arg['flags']
                   or 'tokenizer' in arg['flags']
                   or 'converter' in arg['flags']
                   or 'data_loader' in arg['flags']}
    config_args = flatten(config_args)
    config_args = merge(config_args, train_data_loader_info)
    config_args_tuple = [(key, value) for key, value in config_args.items()]
    config_args_tuple = sorted(config_args_tuple, key=lambda item: item[0])

    trained_data_loader_config = load_json(os.path.join(model_path, cd.JSON_DATA_LOADER_CONFIG_NAME))
    trained_data_loader_type = trained_data_loader_config['type']
    trained_data_loader = DataLoaderFactory.factory(cl_type=trained_data_loader_type)
    trained_data_handle = trained_data_loader.load()

    config_name = '_'.join(['{0}-{1}'.format(name, value) for name, value in config_args_tuple])
    model_base_path = os.path.join(cd.TESTS_DATA_DIR,
                                   trained_data_handle.data_name,
                                   model_type)
    config_id = get_data_config_id(filepath=model_base_path, config=config_name)
    model_test_path = os.path.join(model_base_path, str(config_id))

    # Build converter
    converter_type = cd.MODEL_CONFIG[model_type]['converter']
    converter_args = {key: arg['value'] for key, arg in network_args.items() if 'converter' in arg['flags']}
    converter = TFConverterFactory.factory(converter_type, **converter_args)

    conversion_args = converter.load_conversion_args(filepath=model_test_path, prefix='{0}_{1}'.format(
        test_prefix,
        save_prefix)
    if test_prefix is not None
    else save_prefix)

    label_map = conversion_args['label_map']
    inv_label_map = {class_key: {value: key for key, value in class_value.items()} for class_key, class_value in
                     label_map.items()}

    save_name = model_type
    if repetition_prefix is not None:
        save_name += '_repetition_{}'.format(repetition_prefix)
    if save_prefix is not None:
        save_name += '_key_{}'.format(save_prefix)

    predictions_name = save_name + '_unseen_predictions.json'
    raw_predictions_name = save_name + '_raw_predictions.npy'

    predictions_path = os.path.join(cd.UNSEEN_DATA_DIR, model_type, test_name, predictions_name)
    predictions = load_json(predictions_path)

    raw_predictions_path = os.path.join(cd.UNSEEN_DATA_DIR, model_type, test_name, raw_predictions_name)
    raw_predictions = np.load(raw_predictions_path, allow_pickle=True).item()

    # Apply threshold to filter out predictions
    raw_predictions = {key: np.max(value, axis=-1) for key, value in raw_predictions.items()}

    # Load dataset
    df_path = os.path.join(cd.LOCAL_DATASETS_DIR, 'arg_sci_chat',
                           'papers_model_components_dataset_{}.csv'.format(threshold))
    df = pd.read_csv(df_path)

    return {
        'raw_predictions': raw_predictions,
        'predictions': predictions,
        'df': df,
        'inv_label_map': inv_label_map
    }


def build_components_dataset(token_level_arguments, threshold_value, distance=10):
    components_df = {}

    for doc, doc_args in tqdm(token_level_arguments.items()):
        doc_arguments_amount = len(doc_args)
        for source_idx in range(doc_arguments_amount):
            for target_idx in range(source_idx + 1, doc_arguments_amount):

                difference = target_idx - source_idx

                difference_array = [0] * distance * 2
                if difference > distance:
                    difference_array[-distance:] = [1] * distance
                elif difference < -distance:
                    difference_array[:distance] = [1] * distance
                elif difference > 0:
                    difference_array[-distance: distance + difference] = [1] * difference
                elif difference < 0:
                    difference_array[distance + difference: distance] = [1] * -difference

                components_df.setdefault('Doc', []).append(doc)
                components_df.setdefault('Source Idx', []).append(source_idx)
                components_df.setdefault('Source', []).append(doc_args[source_idx])
                components_df.setdefault('Target Idx', []).append(target_idx)
                components_df.setdefault('Target', []).append(doc_args[target_idx])
                components_df.setdefault('Distance', []).append(str(difference_array))
                components_df.setdefault('Difference', []).append(difference)

    components_df = pd.DataFrame.from_dict(components_df)
    save_path = os.path.join(cd.LOCAL_DATASETS_DIR, 'arg_sci_chat',
                             'papers_model_components_dataset_{}.csv'.format(threshold_value))
    components_df.to_csv(save_path, index=False)

    return components_df.empty


def annotate_corpus(df, raw_predictions, predictions, inv_label_map, threshold):
    df_copy = df.copy(deep=True)

    # Quick fix

    for key, value in predictions.items():
        raw_values = raw_predictions[key]
        invalid_indexes = np.where(raw_values < threshold)[0]
        print('Key - {0} -> Invalid indexes - {1} ({2})'.format(key, len(invalid_indexes),
                                                                len(invalid_indexes) / len(value)))

        value = np.array(
            [inv_label_map[key][item] if inv_label_map[key][item] is not None else 'no-link' for item in value])
        value[invalid_indexes] = 'no-link'

        df_copy[key] = value

    # Save annotated corpus
    df_path = os.path.join(cd.LOCAL_DATASETS_DIR, 'arg_sci_chat',
                           'papers_model_components_dataset_{}.csv'.format(threshold))
    df_copy.to_csv(df_path, index=False)

    return df_copy


def run_component_model(components_model_info, threshold):
    # Update test config file
    config_path = os.path.join(cd.CONFIGS_DIR, cd.JSON_UNSEEN_TEST_CONFIG_NAME)
    config_data = load_json(config_path)

    for key, value in components_model_info.items():
        config_data[key] = value

    save_json(config_path, config_data)

    # Update data loader info
    loader_config_path = os.path.join(cd.CONFIGS_DIR, cd.JSON_DATA_LOADER_CONFIG_NAME)
    loader_config_data = load_json(loader_config_path)

    loader_config_data['configs']['arg_sci_chat_model_components_papers']['threshold'] = threshold
    save_json(loader_config_path, loader_config_data)

    # Run script
    script_path = os.path.join(cd.PROJECT_DIR, 'runnables', 'test_unseen_data.py')
    cmd = ['python', script_path]
    process = Popen(cmd, shell=False)
    process.wait()


def extract_argumentative_links(components_annotated_df):
    doc_links = {}
    doc_links_counter = {}
    per_doc_distribution = {}
    overall_distribution = {
        'link': 0,
        'no-link': 0
    }

    documents = set(components_annotated_df['Doc'].values)

    for document in documents:
        doc_df = components_annotated_df[components_annotated_df['Doc'] == document]

        link_counter = 0
        nolink_counter = 0

        for row_idx, row in doc_df.iterrows():
            source = row['Source']
            target = row['Target']
            relation = row['relation_type']

            if relation == 'link':
                doc_links.setdefault(document, []).append((source, target))
                link_counter += 1
            else:
                nolink_counter += 1

            if document not in doc_links_counter:
                doc_links_counter.setdefault(document, {}).setdefault('link', link_counter)
                doc_links_counter.setdefault(document, {}).setdefault('no-link', nolink_counter)
            else:
                doc_links_counter[document]['link'] = link_counter
                doc_links_counter[document]['no-link'] = nolink_counter

        for key in ['link', 'no-link']:
            per_doc_distribution.setdefault(key, []).append(doc_links_counter[document][key] / doc_df.shape[0])
            overall_distribution[key] += doc_links_counter[document][key]

    per_doc_distribution = {key: np.mean(value) for key, value in per_doc_distribution.items()}
    overall_distribution = {key: value / components_annotated_df.shape[0] for key, value in
                            overall_distribution.items()}

    return doc_links, overall_distribution, per_doc_distribution


# Statistics

def evaluate_arguments_distribution(argument_labels):
    per_doc_distribution = {}

    for document, doc_labels in argument_labels.items():
        doc_counter = Counter(doc_labels)
        doc_labels_amount = len(doc_labels)

        for key, value in doc_counter.items():
            per_doc_distribution.setdefault(key, []).append(value / doc_labels_amount)

    per_doc_distribution = {key: np.mean(value) for key, value in per_doc_distribution.items()}

    all_labels = list(argument_labels.values())
    all_labels = [item for seq in all_labels for item in seq]
    all_counter = Counter(all_labels)
    all_labels_amount = len(all_labels)

    overall_distribution = {key: value / all_labels_amount for key, value in all_counter.items()}

    return overall_distribution, per_doc_distribution


def retrieve_relations(unique_sentences, doc_data):
    relations = []
    relation_types = []
    for sentence in unique_sentences:
        sent_data = doc_data[doc_data['Source'] == sentence]
        for row_idx, row in sent_data.iterrows():
            # Add relation if both source and target are argumentative and if relation exists
            if row['relation_type'] == 'link':
                relations.append((row['Source Idx'], row['Target Idx']))
                relation_types.append(row['relation_type'])

    return relations, relation_types


def retrieve_unique_sentences(doc_data):
    sources = doc_data['Source'].values
    targets = doc_data['Target'].values

    all_sentences = np.concatenate((sources, targets))
    return np.unique(all_sentences)


def get_message_facts(message):
    chunks = message.split(os.linesep)
    fact_section = False
    facts = []
    for chunk in chunks:
        if fact_section and len(chunk.strip()):
            if chunk.startswith('1.') or chunk.startswith('2.'):
                chunk = chunk[3:]
            facts.append(chunk)

        if 'Facts:' in chunk.strip():
            fact_section = True

    return facts


def get_sentences_from_fact(fact, doc_sentences):
    found_sentences = []
    for doc_sent in doc_sentences:
        if doc_sent in fact or fact in doc_sent:
            found_sentences.append(doc_sent)

    return found_sentences


def average_argumentative_facts(annotated_papers_df, summary_df, dialogue_df, documents):
    average_argumentative_facts = []
    for document in tqdm(documents):
        doc_assignments = summary_df[summary_df['Article Title'] == document]['Assignment'].values
        doc_dialogues = dialogue_df[dialogue_df['Assignment'].isin(doc_assignments)]
        de_messages = doc_dialogues[doc_dialogues['Role'] == 'Domain Expert']['Message'].values
        de_facts = [get_message_facts(msg) for msg in de_messages]
        de_facts = [item for seq in de_facts for item in seq]

        if not len(doc_assignments):
            continue

        doc_paper_data = annotated_papers_df[annotated_papers_df['Doc'] == document]
        doc_sentences = retrieve_unique_sentences(doc_paper_data)

        argumentative_facts = []
        for fact in de_facts:
            for doc_sent in doc_sentences:
                if fact in doc_sent or doc_sent in fact:
                    argumentative_facts.append(fact)
                    break

        doc_ratio = len(argumentative_facts) / len(de_facts)
        average_argumentative_facts.append(doc_ratio)

    return np.mean(average_argumentative_facts)


def average_argumentative_linked_facts(annotated_papers_df, summary_df, dialogue_df, documents):
    average_linked_facts = []
    for document in tqdm(documents):
        doc_assignments = summary_df[summary_df['Article Title'] == document]['Assignment'].values
        doc_dialogues = dialogue_df[dialogue_df['Assignment'].isin(doc_assignments)]
        de_messages = doc_dialogues[doc_dialogues['Role'] == 'Domain Expert']['Message'].values
        de_facts = [get_message_facts(msg) for msg in de_messages]
        de_facts = [item for seq in de_facts for item in seq]

        if not len(doc_assignments):
            continue

        doc_paper_data = annotated_papers_df[annotated_papers_df['Doc'] == document]
        doc_sentences = retrieve_unique_sentences(doc_paper_data)

        linked_facts = 0
        fact_to_sentence_map = {fact: get_sentences_from_fact(fact, doc_sentences) for fact in de_facts}
        for (fact_source, fact_target) in combinations(de_facts, 2):
            source_sentences = fact_to_sentence_map[fact_source]
            target_sentences = fact_to_sentence_map[fact_target]

            can_skip = False
            for source_sent in source_sentences:
                for target_sent in target_sentences:
                    corresponding_arg_data = annotated_papers_df[
                        (annotated_papers_df['Source'] == source_sent) & (
                                annotated_papers_df['Target'] == target_sent) & (
                                annotated_papers_df['Doc'] == document)]
                    if not corresponding_arg_data.empty:
                        pair_relation = corresponding_arg_data['relation_type'].values[0]
                        if pair_relation is not None and pair_relation != 'None' and pair_relation is not np.nan:
                            linked_facts += 1
                            can_skip = True
                            break

                if can_skip:
                    break

        average_linked_facts.append(linked_facts / len(de_facts))

    return np.mean(average_linked_facts)


def get_unlabelled_nodes(arguments, unlabelled_text, documents):
    unlabelled_nodes_ratio = []
    for document in tqdm(documents):

        if document in arguments:
            doc_arguments = len(arguments[document])
        else:
            doc_arguments = 0

        if document in unlabelled_text:
            doc_unlabelled = len(unlabelled_text[document])
        else:
            doc_unlabelled = 0

        unlabelled_nodes_ratio.append(doc_unlabelled / (doc_arguments + doc_unlabelled))

    return np.mean(unlabelled_nodes_ratio)


def get_links_ratios(annotated_papers_df, documents):
    connected_nodes_ratio = []
    connected_pairs_ratio = []
    for document in tqdm(documents):
        doc_paper_data = annotated_papers_df[annotated_papers_df['Doc'] == document]
        doc_sentences = retrieve_unique_sentences(doc_paper_data)
        doc_relations, relation_types = retrieve_relations(doc_sentences, doc_paper_data)

        nodes_with_link = len(set([item for pair in doc_relations for item in pair]))
        nodes = max(len(doc_sentences), 1)
        pairs = len(doc_relations)
        possible_pairs = max(len(list(combinations(doc_sentences, 2))), 1)

        connected_nodes_ratio.append(nodes_with_link / float(nodes))
        connected_pairs_ratio.append(pairs / float(possible_pairs))

    return np.mean(connected_nodes_ratio), np.mean(connected_pairs_ratio)


def plot_metrics(threshold_values, metrics, metric_names, show_together=False):
    if not show_together:
        fig, axs = plt.subplots(len(metrics), 1)
        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            axs[idx].plot(threshold_values, metric, linestyle='dotted', marker='x', markersize=MEDIUM_SIZE,
                          linewidth=SMALL_SIZE)
            axs[idx].legend([metric_name])

            axs[idx].xlabel('Threshold')
    else:
        fig, axs = plt.subplots(1, 1)
        for metric, marker in zip(metrics, plot_markers):
            if type(metric) not in [list, np.ndarray]:
                axs.plot(threshold_values, len(threshold_values) * [metric], linestyle='dotted', linewidth=SMALL_SIZE)
            else:
                axs.plot(threshold_values, metric, linestyle='dotted', marker=marker, markersize=MEDIUM_SIZE,
                         linewidth=SMALL_SIZE)

        axs.set_xlabel(r'$\delta$')
        axs.legend(metric_names)


# Visualization

def plot_doc_graph(source_to_label_map, doc_relations, relation_types):
    G = nx.DiGraph()
    G.add_edges_from(doc_relations)

    node_colour_map = {
        'None': 0.5,
        'own_claim': 0.0,
        'background_claim': 1.0
    }

    values = [node_colour_map.get(source_to_label_map.get(node, 'None'), 0.0) for node in G.nodes()]

    relation_types = np.array(relation_types)
    doc_relations = np.array(doc_relations)

    support_edges_indexes = np.where(relation_types == 'supports')[0]
    support_edges = doc_relations[support_edges_indexes]

    attack_edges_indexes = np.where(relation_types == 'contradicts')[0]
    attack_edges = doc_relations[attack_edges_indexes]

    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('PiYG'), node_color=values, node_size=250)
    nx.draw_networkx_labels(G, pos)
    if len(support_edges):
        nx.draw_networkx_edges(G, pos, edgelist=support_edges, edge_color='g', arrows=True)

    if len(attack_edges):
        nx.draw_networkx_edges(G, pos, edgelist=attack_edges, edge_color='r', arrows=True)
    plt.show()


if __name__ == '__main__':

    # Limits gpu usage and restricts gpu visibility
    limit_gpu_usage(limit_gpu_visibility=True, gpu_end_index=0, gpu_start_index=1)

    # Settings

    # How many threshold values do you want to check? Plots will take into account all values
    threshold_values = np.arange(0.55, 0.95, 0.05)
    # threshold_values = [0.75]

    # Whether to save and show statistics
    plot_data = False

    # Tokens model
    # Note: change 'test_name' and possibly 'model_type' if you want to test your own model
    tokens_model_info = {
        'model_type': 'drinventor_tf_tokens_scibert_crf',
        'test_name': 'test_scibert_crf',
        'data_loader_type': 'arg_sci_chat_tokens_papers',
        'evaluation_folder': cd.TRAIN_AND_TEST_DIR,
        'save_prefix': None,
        'test_prefix': None,
        'repetition_prefix': '0',
    }

    is_bert_model = 'bert' in tokens_model_info['model_type']

    # Components model
    # Note: change 'test_name' and possibly 'model_type' if you want to test your own model
    components_model_info = {
        'model_type': 'drinventor_tf_components_scibert',
        'test_name': 'test_scibert',
        'data_loader_type': 'arg_sci_chat_model_components_papers',
        'training_test': 'train_and_test',
        'save_prefix': None,
        'test_prefix': None,
        'repetition_prefix': '0',
    }

    # ----

    dialogue_df = pd.read_csv(os.path.join(cd.LOCAL_DATASETS_DIR, 'arg_sci_chat', 'dialogue_df.csv'))
    summary_df = pd.read_csv(os.path.join(cd.LOCAL_DATASETS_DIR, 'arg_sci_chat', 'summary_df.csv'))

    print("Dataset size -> ", dialogue_df.shape)

    base_plot_path = os.path.join(cd.PROJECT_DIR, 'plot_data')

    if not os.path.isdir(base_plot_path):
        os.makedirs(base_plot_path)

    if not len(os.listdir(base_plot_path)) or not plot_data:

        # Tokens

        print('Loading tokens annotation data...')
        # Load tokens model
        # Get dataset and model predictions
        # raw_predictions, predictions, df, inv_label_map
        tokens_model_data = load_tokens_annotation_data(**tokens_model_info)
        print('Loading completed!')

        # Graph measures
        total_connected_nodes_ratio = []
        total_connected_pairs_ratio = []
        total_unlabelled_nodes_ratio = []

        ref_all_arg_distr, ref_per_doc_arg_distr, \
        ref_all_links_distr, ref_per_doc_links_distr = determine_drinventor_graph_measures()

        total_all_arg_distr = {}
        total_per_doc_arg_distr = {}
        total_all_links_distr = {}
        total_per_doc_links_distr = {}

        # Arg measures
        total_avg_arg_facts = []
        total_avg_arg_links = []

        for threshold in threshold_values:
            print('Considering threshold = ', threshold)

            print('Annotating ArgSciChat papers (tokens version)...')
            # Annotate tokens corpus using given threshold
            tokens_annotated_df = annotate_tokens_corpus(**tokens_model_data,
                                                         threshold=threshold,
                                                         null_label='O',
                                                         is_bert_model=is_bert_model)

            print('Extracting token level arguments...')
            # Extract arguments
            token_level_arguments, argument_labels, unlabelled_doc_text = extract_token_level_arguments(
                tokens_annotated_df)

            print('Building component level arguments dataset...')
            # Build components dataset
            empty_dataset = build_components_dataset(token_level_arguments, threshold_value=threshold)

            if empty_dataset:
                print('Component dataset is empty!')

                for key in doc_links_distr:
                    total_per_doc_links_distr.setdefault(key, []).append(0)
                    total_all_links_distr.setdefault(key, []).append(0)

                for key in doc_arg_distr:
                    total_per_doc_arg_distr.setdefault(key, []).append(0)
                    total_all_arg_distr.setdefault(key, []).append(0)

                total_unlabelled_nodes_ratio.append(0)
                total_connected_nodes_ratio.append(0)
                total_connected_pairs_ratio.append(0)

                total_avg_arg_facts.append(0)
                total_avg_arg_links.append(0)
            else:
                print('Running component model...')
                # Run component model on argument dataset
                run_component_model(components_model_info, threshold=threshold)

                print('Loading component model predictions...')
                # Load component model predictions
                components_model_data = load_components_annotation_data(**components_model_info, threshold=threshold)

                # Annotate components corpus using given threshold
                components_annotated_df = annotate_corpus(**components_model_data,
                                                          threshold=threshold)
                documents = set(components_annotated_df['Doc'].values)
                documents = [item for item in documents if not summary_df[summary_df['Article Title'] == item].empty]

                # Extract argumentative links
                document_links, doc_links_distr, all_links_distr = extract_argumentative_links(components_annotated_df)

                for key in doc_links_distr:
                    total_per_doc_links_distr.setdefault(key, []).append(doc_links_distr[key])
                    total_all_links_distr.setdefault(key, []).append(all_links_distr[key])

                # Compute statistics
                doc_arg_distr, all_arg_distr = evaluate_arguments_distribution(argument_labels)

                for key in doc_arg_distr:
                    total_per_doc_arg_distr.setdefault(key, []).append(doc_arg_distr[key])
                    total_all_arg_distr.setdefault(key, []).append(all_arg_distr[key])

                # Graph Measures
                unlabelled_nodes_ratio = get_unlabelled_nodes(arguments=token_level_arguments,
                                                              unlabelled_text=unlabelled_doc_text,
                                                              documents=documents)
                total_unlabelled_nodes_ratio.append(unlabelled_nodes_ratio)

                connected_nodes_ratio, connected_pairs_ratio = get_links_ratios(
                    annotated_papers_df=components_annotated_df,
                    documents=documents)

                total_connected_nodes_ratio.append(connected_nodes_ratio)
                total_connected_pairs_ratio.append(connected_pairs_ratio)

                # Argumentative Measures
                average_argumentative_facts_per_dialogue = average_argumentative_facts(
                    annotated_papers_df=components_annotated_df,
                    summary_df=summary_df,
                    dialogue_df=dialogue_df,
                    documents=documents)
                print('Average argumentative facts per dialogue: ', average_argumentative_facts_per_dialogue)

                average_argumentative_links_per_dialogue = average_argumentative_linked_facts(
                    annotated_papers_df=components_annotated_df,
                    summary_df=summary_df,
                    dialogue_df=dialogue_df,
                    documents=documents)
                print('Average argumentative links per dialogue: ', average_argumentative_links_per_dialogue)

                total_avg_arg_facts.append(average_argumentative_facts_per_dialogue)
                total_avg_arg_links.append(average_argumentative_links_per_dialogue)

        if plot_data:
            # Graph measures
            np.save(os.path.join(base_plot_path, 'total_unlabelled_nodes_ratio'), total_unlabelled_nodes_ratio)
            np.save(os.path.join(base_plot_path, 'total_connected_nodes_ratio'), total_connected_nodes_ratio)
            np.save(os.path.join(base_plot_path, 'total_connected_pairs_ratio'), total_connected_pairs_ratio)

            # Reference measures
            np.save(os.path.join(base_plot_path, 'total_all_arg_distr'), total_all_arg_distr)
            np.save(os.path.join(base_plot_path, 'ref_all_arg_distr'), ref_all_arg_distr)
            np.save(os.path.join(base_plot_path, 'total_per_doc_arg_distr'), total_per_doc_arg_distr)
            np.save(os.path.join(base_plot_path, 'ref_per_doc_arg_distr'), ref_per_doc_arg_distr)

            np.save(os.path.join(base_plot_path, 'total_all_links_distr'), total_all_links_distr)
            np.save(os.path.join(base_plot_path, 'ref_all_links_distr'), ref_all_links_distr)
            np.save(os.path.join(base_plot_path, 'total_per_doc_links_distr'), total_per_doc_links_distr)
            np.save(os.path.join(base_plot_path, 'ref_per_doc_links_distr'), ref_per_doc_links_distr)

            # Argumentative measures
            np.save(os.path.join(base_plot_path, 'total_avg_arg_facts'), total_avg_arg_facts)
            np.save(os.path.join(base_plot_path, 'total_avg_arg_links'), total_avg_arg_links)
    else:
        if plot_data:
            total_unlabelled_nodes_ratio = np.load(os.path.join(base_plot_path, 'total_unlabelled_nodes_ratio.npy'))
            total_connected_nodes_ratio = np.load(os.path.join(base_plot_path, 'total_connected_nodes_ratio.npy'))
            total_connected_pairs_ratio = np.load(os.path.join(base_plot_path, 'total_connected_pairs_ratio.npy'))

            total_all_arg_distr = np.load(os.path.join(base_plot_path, 'total_all_arg_distr.npy'),
                                          allow_pickle=True).item()
            ref_all_arg_distr = np.load(os.path.join(base_plot_path, 'ref_all_arg_distr.npy'), allow_pickle=True).item()
            total_per_doc_arg_distr = np.load(os.path.join(base_plot_path, 'total_per_doc_arg_distr.npy'),
                                              allow_pickle=True).item()
            ref_per_doc_arg_distr = np.load(os.path.join(base_plot_path, 'ref_per_doc_arg_distr.npy'),
                                            allow_pickle=True).item()

            total_all_links_distr = np.load(os.path.join(base_plot_path, 'total_all_links_distr.npy'),
                                            allow_pickle=True).item()
            ref_all_links_distr = np.load(os.path.join(base_plot_path, 'ref_all_links_distr.npy'),
                                          allow_pickle=True).item()
            total_per_doc_links_distr = np.load(os.path.join(base_plot_path, 'total_per_doc_links_distr.npy'),
                                                allow_pickle=True).item()
            ref_per_doc_links_distr = np.load(os.path.join(base_plot_path, 'ref_per_doc_links_distr.npy'),
                                              allow_pickle=True).item()

            total_avg_arg_facts = np.load(os.path.join(base_plot_path, 'total_avg_arg_facts.npy'))
            total_avg_arg_links = np.load(os.path.join(base_plot_path, 'total_avg_arg_links.npy'))

    if plot_data:
        # Graph Plots
        plot_metrics(threshold_values=threshold_values,
                     metrics=[total_unlabelled_nodes_ratio, total_connected_nodes_ratio, total_connected_pairs_ratio,
                              total_avg_arg_facts, total_avg_arg_links],
                     metric_names=['Non-Arg Nodes', 'Linked Nodes', 'Linked Pairs', 'Arg Facts', 'Arg Links'],
                     show_together=True)

        # Reference Plots (DrInventor)
        arg_distr_data = []
        arg_distr_names = []
        for key in total_all_arg_distr:
            arg_distr_data.append(total_all_arg_distr[key])
            arg_distr_names.append('ArgSciChat - ' + key)
            arg_distr_data.append(ref_all_arg_distr[key])
            arg_distr_names.append('DrInventor - ' + key)

        plot_metrics(threshold_values=threshold_values,
                     metrics=arg_distr_data,
                     metric_names=arg_distr_names,
                     show_together=True)

        arg_distr_data = []
        arg_distr_names = []
        for key in total_all_links_distr:
            arg_distr_data.append(total_all_links_distr[key])
            arg_distr_names.append('ArgSciChat - ' + key)
            arg_distr_data.append(ref_all_links_distr[key])
            arg_distr_names.append('DrInventor - ' + key)

        plot_metrics(threshold_values=threshold_values,
                     metrics=arg_distr_data,
                     metric_names=arg_distr_names,
                     show_together=True)

        plt.show()
