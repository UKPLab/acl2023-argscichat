"""

Compute and add the argumentative mask corresponding to each paper content.
The argumentative mask is a binary vector: 1 if sentence is argumentative and 0 otherwise.

It can be used by the LED model to identify arguments while training for DE reply generation.

"""

import os
import argscichat_allennlp.const_define as cd
from argscichat_allennlp.utility.json_utils import load_json, save_json
import pandas as pd
from nltk import sent_tokenize
import string


def punctuation_filtering(line):
    """
    Filters given sentences by removing punctuation
    """

    table = str.maketrans('', '', string.punctuation)
    trans = [w.translate(table) for w in line.split()]

    return ' '.join([w for w in trans if w != ''])


if __name__ == '__main__':

    # Settings

    threshold = 0.75
    sentence_dataset = True

    # ----

    if sentence_dataset:
        prefix = 'sentence_'
    else:
        prefix = ''

    base_path = os.path.join(cd.PROJECT_DIR, 'argscichat_train_dev')
    argument_df = pd.read_csv(os.path.join(base_path, 'papers_model_components_dataset_{}.csv'.format(threshold)))
    train_data = load_json(os.path.join(base_path, '{}train.json'.format(prefix)))
    val_data = load_json(os.path.join(base_path, '{}val.json'.format(prefix)))

    for data, data_name in zip([train_data, val_data], ['train', 'val']):
        for article, article_data in data.items():
            article_content = article_data['content']
            article_arguments = argument_df[argument_df['Doc'] == article]
            article_arguments = article_arguments['Source'].values.tolist() + article_arguments['Target'].values.tolist()
            article_arguments = list(set(article_arguments))
            article_argument_sentences = [sent_tokenize(arg) for arg in article_arguments]
            article_argument_sentences = [item for seq in article_argument_sentences
                                          for item in seq if len(punctuation_filtering(item.strip()))]

            argument_mask = [0] * len(article_content)
            found_arguments = []
            for sent_idx, article_sentence in enumerate(article_content):
                article_sentence = punctuation_filtering(article_sentence).replace(' ', '')
                for arg_sent in article_argument_sentences:
                    filt_arg_sent = punctuation_filtering(arg_sent).replace(' ', '')
                    if arg_sent in found_arguments:
                        continue
                    if filt_arg_sent in article_sentence or article_sentence in filt_arg_sent:
                        found_arguments.append(arg_sent)
                        argument_mask[sent_idx] = 1

            remaining = set(article_argument_sentences).difference(set(found_arguments))
            if remaining:
                print('Missing arguments: {}'.format(remaining))
            article_data['argument_mask_{}'.format(threshold)] = argument_mask
            assert len(argument_mask) == len(article_content)

        save_json(os.path.join(base_path, '{0}{1}.json'.format(prefix, data_name)), data)
