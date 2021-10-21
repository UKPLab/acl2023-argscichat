#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from datetime import datetime

from mephisto.abstractions.databases.local_database import LocalMephistoDB
from mephisto.data_model.worker import Worker
from mephisto.tools.data_browser import DataBrowser as MephistoDataBrowser
import numpy as np
import collection_tool.const_define as cd
import os
import pandas as pd

db = LocalMephistoDB()
mephisto_data_browser = MephistoDataBrowser(db=db)

task_name = "name_of_mephisto_task"
suffix_name = "first"
tasks_to_show = "a"
save_results = True

units = mephisto_data_browser.get_units_for_task_name(task_name)

if tasks_to_show in ["all", "a"]:
    DO_REVIEW = False
else:
    units = [u for u in units if u.get_status() == "completed"]
    print(
        "You will be reviewing actual tasks with this flow. Tasks that you either Accept or Pass "
        "will be paid out to the worker, while rejected tasks will not. Passed tasks will be "
        "specially marked such that you can leave them out of your dataset. \n"
        "When you pass on a task, the script gives you an option to disqualify the worker "
        "from future tasks by assigning a qualification. If provided, this worker will no "
        "longer be able to work on tasks where the set --block-qualification shares the same name.\n"
        "You should only reject tasks when it is clear the worker has acted in bad faith, and "
        "didn't actually do the task. Prefer to pass on tasks that were misunderstandings."
    )


def format_for_printing_data(data):
    # Custom tasks can define methods for how to display their data in a relevant way
    worker_name = Worker(db, data["worker_id"]).worker_name
    contents = data["data"]

    if contents['save_data'] is not None:
        article_abstract = contents['save_data'][0]['abstract']
        article_title = contents['save_data'][0]['title']

        start_date = contents['save_data'][0]['start_time']
        start_date = datetime.strptime(start_date, '%d-%m-%Y-%H-%M-%S')

        end_date = contents['save_data'][-1]['end_time']
        end_date = datetime.strptime(end_date, '%d-%m-%Y-%H-%M-%S')

        duration = end_date - start_date
        duration = duration.total_seconds()
    else:
        # Timeout
        first_msg = contents['messages'][0]
        last_msg = contents['messages'][-1]

        article_abstract = first_msg['abstract'] if 'abstract' in first_msg else None
        article_title = first_msg['title'] if 'title' in first_msg else None

        start_date = first_msg['time'] if 'time' in first_msg else None
        if start_date is not None:
            start_date = datetime.strptime(start_date, '%d-%m-%Y-%H-%M-%S')

        end_date = last_msg['time'] if 'time' in last_msg else None
        if end_date is not None:
            end_date = datetime.strptime(end_date, '%d-%m-%Y-%H-%M-%S')

        if start_date is not None and end_date is not None:
            duration = end_date - start_date
            duration = duration.total_seconds()
        else:
            duration = None

    agent_role = contents['agent_name']

    metadata_string = (
        f"Worker: {worker_name}\nUnit: {data['unit_id']}\n"
        f"Assignment: {data['assignment_id']}\nStatus: {data['status']}\n"
        f"Duration (seconds): {int(duration) if duration is not None else duration}"
        f" -- Duration (minutes): {duration / 60. if duration is not None else duration}\n"
        f"Agent name: {agent_role}\n"
    )

    contents_string = (
        f"Article Title: {article_title}\n"
        f"Article Abstract: \n{article_abstract}\n"
    )

    message_string = ""
    for msg_idx, msg in enumerate(contents['messages']):

        # First message is a system message (loading article...)
        if msg_idx == 0:
            continue

        if 'id' not in msg:
            continue

        if 'text' not in msg:
            continue

        msg_role = msg['id']
        msg_text = msg['text']
        if msg_idx == len(contents['messages']) - 1:
            msg_flag = True
        else:
            msg_flag = False
        msg_id = msg['message_id']
        message_string += f"[{msg_role}] {msg_text}\nEpisode_done: {msg_flag}\nID: {msg_id}\n"
        message_string += '*' * 50 + '\n'

    return f"-------------------\n{metadata_string}\n{contents_string}\n{message_string}-------------------\n"


def parse_unit_data(data):
    worker_name = Worker(db, data["worker_id"]).worker_name
    contents = data["data"]

    if contents['save_data'] is None and not contents['messages']:
        print("[No data] Skipping worker: {}".format(worker_name))
        return None, None, None

    # Good task
    if contents['save_data'] is not None:
        article_abstract = contents['save_data'][0]['abstract']
        article_title = contents['save_data'][0]['title']
        article_introduction = contents['save_data'][0]['introduction']

        start_date = contents['save_data'][0]['start_time']
        start_date = datetime.strptime(start_date, '%d-%m-%Y-%H-%M-%S')

        end_date = contents['save_data'][-1]['end_time']
        end_date = datetime.strptime(end_date, '%d-%m-%Y-%H-%M-%S')

        duration = end_date - start_date
        duration = duration.total_seconds()
    else:
        # Timeout
        first_msg = contents['messages'][0]
        last_msg = contents['messages'][-1]

        article_abstract = first_msg['abstract'] if 'abstract' in first_msg else None
        article_title = first_msg['title'] if 'title' in first_msg else None
        article_introduction = first_msg['introduction'] if 'introduction' in first_msg else None

        start_date = first_msg['time'] if 'time' in first_msg else None
        if start_date is not None:
            start_date = datetime.strptime(start_date, '%d-%m-%Y-%H-%M-%S')

        end_date = last_msg['time'] if 'time' in last_msg else None
        if end_date is not None:
            end_date = datetime.strptime(end_date, '%d-%m-%Y-%H-%M-%S')

        if start_date is not None and end_date is not None:
            duration = end_date - start_date
            duration = duration.total_seconds()
        else:
            duration = None

    agent_role = contents['agent_name']

    worker_data = {
        'Worker_name': worker_name,
        'Unit': data['unit_id'],
        'Assignment': data['assignment_id'],
        'Status': data['status'] if contents['save_data'] is not None else 'incomplete',
        'Duration (seconds)': int(duration) if duration is not None else duration,
        'Duration (minutes)': '{:.2f}'.format(duration / 60) if duration is not None else duration,
        'Agent_role': agent_role
    }

    article_data = {
        'Title': article_title,
        'Abstract': article_abstract,
        'Introduction': article_introduction
    }

    dialogue_data = {}
    for msg_idx, msg in enumerate(contents['messages']):

        # First message is a system message (loading article...)
        if msg_idx == 0:
            continue

        if 'id' not in msg:
            continue

        if 'text' not in msg:
            continue

        msg_role = msg['id']
        msg_text = msg['text']
        if msg_idx == len(contents['messages']) - 1:
            msg_flag = True
        else:
            msg_flag = False
        msg_id = msg['message_id']

        dialogue_data.setdefault('Role', []).append(msg_role)
        dialogue_data.setdefault('Episode_done', []).append(msg_flag)
        dialogue_data.setdefault('ID', []).append(msg_id)
        dialogue_data.setdefault('Text', []).append(msg_text)

    return worker_data, article_data, dialogue_data


def get_messages_with_filter(message_data, filter_func):
    return [item for item in message_data if filter_func(item.split('Fact')[0])]


def get_messages_with_facts(message_data):
    return [(item, 1 if '2.' not in item else 2) for item in message_data if 'Facts:' in item]


def get_average_fact_usage(message_data):
    facts_info = get_messages_with_facts(message_data)
    counts = [item[1] for item in facts_info]

    return np.mean(counts)


"""
Saving formats:

1. csv containing -> worker_data, article_title, interactions, questions, statements, messages_with_facts, avg_fact_number
2. csv containing -> worker_name, assignment_id, unit_id, message, message_number

"""

disqualification_name = "scientific-discussion-block-qualification"
csv_data_dict = {}
dialogue_data_dict = {}
for unit in units:
    try:
        unit_data = mephisto_data_browser.get_data_from_unit(unit)
    except IndexError:
        continue
    worker_data, article_data, dialogue_data = parse_unit_data(unit_data)
    if worker_data is not None:
        print(format_for_printing_data(unit_data))

    # Build csv data
    if worker_data is None:
        continue

    # Worker data
    for key, value in worker_data.items():
        csv_data_dict.setdefault(key, []).append(value)

    # Article data
    csv_data_dict.setdefault('Article Title', []).append(article_data['Title'])

    if worker_data['Status'] != 'completed':
        try:
            interactions = int(len(dialogue_data['Text']) / 2)
        except Exception:
            interactions = None

        try:
            questions = len(get_messages_with_filter(dialogue_data['Text'], lambda item: item.strip().endswith('?')))
        except Exception:
            questions = None

        try:
            statements = len(get_messages_with_filter(dialogue_data['Text'], lambda item: not item.strip().endswith('?')))
        except Exception:
            statements = None

        try:
            supported_messages = len(get_messages_with_facts(dialogue_data['Text']))
        except Exception:
            supported_messages = None

        try:
            avg_fact_usage = get_average_fact_usage(dialogue_data['Text'])
        except Exception:
            avg_fact_usage = None
    else:
        interactions = int(len(dialogue_data['Text']) / 2)
        questions = len(get_messages_with_filter(dialogue_data['Text'], lambda item: item.strip().endswith('?')))
        statements = len(get_messages_with_filter(dialogue_data['Text'], lambda item: not item.strip().endswith('?')))
        supported_messages = len(get_messages_with_facts(dialogue_data['Text']))
        avg_fact_usage = get_average_fact_usage(dialogue_data['Text'])

    # Dialogue stats
    csv_data_dict.setdefault('Interactions', []).append(interactions)
    csv_data_dict.setdefault('Questions', []).append(questions)
    csv_data_dict.setdefault('Statements', []).append(statements)
    csv_data_dict.setdefault('Supported messages', []).append(supported_messages)
    csv_data_dict.setdefault('Average fact usage', []).append(avg_fact_usage)

    # Build dialogue dataset (ParlAI format)
    if worker_data['Status'] == 'completed':
        total_messages = len(dialogue_data['Text'])
        for msg_idx in range(total_messages):
            # Just consider worker's messages
            if dialogue_data['Role'][msg_idx] == worker_data['Agent_role']:
                dialogue_data_dict.setdefault('Worker_name', []).append(worker_data['Worker_name'])
                dialogue_data_dict.setdefault('Assignment', []).append(worker_data['Assignment'])
                dialogue_data_dict.setdefault('Unit', []).append(worker_data['Unit'])
                dialogue_data_dict.setdefault('Role', []).append(dialogue_data['Role'][msg_idx])
                dialogue_data_dict.setdefault('Message', []).append(dialogue_data['Text'][msg_idx])
                dialogue_data_dict.setdefault('Message Number', []).append(msg_idx)

# Saving
if save_results and len(csv_data_dict):
    # csv data
    csv_data = pd.DataFrame.from_dict(csv_data_dict)
    csv_save_path = os.path.join(cd.LOCAL_DATASETS_DIR, 'argscichat_collection', '{0}_{1}.csv'.format(task_name, suffix_name))
    csv_data.to_csv(csv_save_path, index=False)

    # dialogue dataset
    dialogue_csv_data = pd.DataFrame.from_dict(dialogue_data_dict)
    dialogue_csv_save_path = os.path.join(cd.LOCAL_DATASETS_DIR, 'argscichat_collection', '{0}_{1}_dialogues.csv'.format(task_name, suffix_name))
    dialogue_csv_data.to_csv(dialogue_csv_save_path, index=False)
