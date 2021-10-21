"""

Converts ArgSciChat corpus format into QASPER format for Allen NLP.

"""

import os
import pandas as pd
from nltk import sent_tokenize
import argscichat_allennlp.const_define as cd
import string
from ast import literal_eval
from argscichat_allennlp.utility.json_utils import save_json


def punctuation_filtering(line):
    """
    Filters given sentences by removing punctuation
    """

    table = str.maketrans('', '', string.punctuation)
    trans = [w.translate(table) for w in line.split()]

    return ' '.join([w for w in trans if w != ''])


if __name__ == '__main__':

    df_path = os.path.join(cd.PROJECT_DIR, 'argscichat_train_dev', 'argscichat_corpus.csv')
    df = pd.read_csv(df_path)

    grouped_df = df.groupby('Article Title')
    train_dict = {}
    val_dict = {}
    for groupd_idx, group in grouped_df:
        article_title = group['Article Title'].values[0]
        article_content = group['Article Content'].values[0]
        content_sentences = sent_tokenize(article_content)
        content_sentences = [sent for sent in content_sentences if len(punctuation_filtering(sent.strip()))]

        message_pairs = []
        for row_idx, row in group.iterrows():
            row_dict = {}
            row_dict['id'] = '{0}_{1}'.format(groupd_idx, row_idx)
            row_dict['P_Message'] = row['P_Message']
            row_dict['DE_Message'] = row['DE_Message']

            row_history = row['Chat History']
            if type(row_history) != float:
                history_sentences = sent_tokenize(row_history)
                history_sentences = [sent for sent in history_sentences if len(punctuation_filtering(sent.strip()))]
                row_dict['history'] = history_sentences
            else:
                row_dict['history'] = []

            row_facts = literal_eval(row['Supporting Facts'])
            facts_sentences = [sent_tokenize(fact) for fact in row_facts]
            facts_sentences = [item for seq in facts_sentences for item in seq if
                               len(punctuation_filtering(item.strip()))]
            row_dict['facts'] = facts_sentences

            message_pairs.append(row_dict)

        split = group['Split'].values[0]
        if split == 'train':
            train_dict.setdefault(article_title, {
                'id': article_title,
                'title': article_title,
                'content': content_sentences,
                'message_pairs': message_pairs
            })
        else:
            val_dict.setdefault(article_title, {
                'id': article_title,
                'title': article_title,
                'content': content_sentences,
                'message_pairs': message_pairs
            })

    save_path = os.path.join(cd.PROJECT_DIR, 'argscichat_train_dev', '{}.json')
    for split_name, data in zip(['train', 'val'], [train_dict, val_dict]):
        save_json(save_path.format(split_name), data)
