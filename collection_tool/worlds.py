# !/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.crowdsourcing.utils.worlds import CrowdOnboardWorld, CrowdTaskWorld
from parlai.core.worlds import validate
from joblib import Parallel, delayed
import numpy as np
from datetime import datetime
import time
import pytz
from datetime import datetime, timedelta
from mysql.connector import connect, Error
import pandas as pd
import collection_tool.const_define as cd
import os


class MultiAgentDialogOnboardWorld(CrowdOnboardWorld):
    def __init__(self, opt, agent):
        super().__init__(opt, agent)
        self.opt = opt
        self.dataset = opt['dataset']

    def parley(self):
        self.agent.agent_id = self.agent.mephisto_agent.get_worker().worker_name
        # self.codes = list(set(self.dataset['Author'].values))
        self.agent.act()
        self.agent.observe({
                "id": "System",
                "text": "Please wait while "
                "we match you with another worker...",
                "episode_done": True,
            })
        self.episodeDone = True


class MultiAgentDialogWorld(CrowdTaskWorld):
    """
    Basic world where each agent gets a turn in a round-robin fashion, receiving as
    input the actions of all other agents since that agent last acted.
    """

    def __init__(self, opt, agents=None, shared=None):
        super().__init__(opt, agents)

        self.dataset = opt['dataset']

        self.event_id = opt['event_id']
        self.db_admin = opt['db_admin']
        self.db_name = opt['db_name']
        self.db_password = opt['db_password']
        self.db_host = opt['db_host']

        # Add passed in agents directly.
        self.agents = agents
        self.acts = [None] * len(agents)
        self.episodeDone = False
        self.max_turns = opt.get("num_turns", 15)
        self.min_turns = opt.get("min_num_turns", 10)
        self.current_turns = 0
        self.send_task_data = opt.get("send_task_data", False)
        self.turn_timeout = opt.get('turn_timeout', 240)
        self.opt = opt

        # Setting up agents' name
        assert len(self.agents) == 2
        self.proponent_idx = 0
        self.domain_expert_idx = 1

        self.agents[self.proponent_idx].agent_id = "Proponent"
        self.agents[self.domain_expert_idx].agent_id = "Domain Expert"

        self.abstract_shown = False
        self.abstract = None
        self.title = None
        self.introduction = None

        # Time limit
        self.time_limit_apply = True
        self.time_limit_start = None
        self.time_limit_first_notification = opt.get("first_notification", 900)
        self.time_limit_first_notification_sent = False
        self.time_limit_second_notification = opt.get("second_notification", 1140)
        self.time_limit_second_notification_sent = False
        self.time_limit_end = opt.get("time_limit", 1200)
        self.has_timed_out = False

        # Internal state to save
        self.conversation_data = []

        # Random messages encouraging opinion exchange
        self.random_tips = 2
        self.tips = [
            "Don't forget to to ask for your partner's opinion or to express your own."
            " You may combine multiple actions into a single message to do that!",
            "Don't forget to to ask for your partner's opinion or to express your own."
            " You may combine multiple actions into a single message to do that!",
        ]
        self.tips = ['[HINT] ' + item for item in self.tips]
        self.minimum_tip_wait = 3
        self.tip_wait_range = 3
        self.random_tip_points = []
        self.current_tip_index = 0

        current_minimum = self.minimum_tip_wait
        current_maximum = current_minimum + self.tip_wait_range
        for _ in range(self.random_tips):
            current_tip_point = np.random.randint(low=current_minimum, high=current_maximum, size=1)
            self.random_tip_points.append(current_tip_point)

            current_minimum = current_maximum
            current_maximum += self.tip_wait_range

        # Timezone
        self.timezone_str = "Europe/Rome"
        self.timezone = pytz.timezone(self.timezone_str)

        # Workers data
        self.workerIds = []
        self.session_idx = None

    def read_from_db(self, codes):

        try:
            with connect(
                    host=self.db_host,
                    user=self.db_admin,
                    password=self.db_password,
                    database=self.db_name
            ) as connection:
                query_str = "SELECT code,proponent_sessions,expert_sessions from participants where event = %s AND code in ({0})".format(
                    ','.join('%s' for _ in codes))
                with connection.cursor(buffered=True) as cursor:
                    cursor.execute(query_str, [self.event_id, *codes])
                    result = cursor.fetchall()
        except Error as e:
            raise e

        print("Read from db -> ", result)
        return {item[0].upper(): {'proponent_sessions': item[1].split(';'), 'expert_sessions': item[2].split(';')}
                for item in result}

    def dummy_determine_participant_roles(self):

        # Always force DE to be the first one
        self.domain_expert_idx = 0
        self.proponent_idx = 1

        firstID = self.agents[0].mephisto_agent.get_worker().worker_name.strip()
        secondID = self.agents[1].mephisto_agent.get_worker().worker_name.strip()
        self.workerIds = [firstID, secondID]
        self.session_idx = 0

    def determine_participants_roles(self):

        # Step 1: get current time
        current_date = datetime.today()
        current_date = self.timezone.localize(current_date)

        # Step 2: Get workers' IDs
        # Proponent_idx -> 0
        # Domain_expert_idx -> 1
        firstID = self.agents[0].mephisto_agent.get_worker().worker_name.upper()
        secondID = self.agents[1].mephisto_agent.get_worker().worker_name.upper()
        self.workerIds = [firstID, secondID]
        print("Worker ids -> ", self.workerIds)

        # Step 3: Retrieve sessions from DB
        codes = np.array([firstID, secondID])
        workers_sessions = self.read_from_db(codes=codes)
        print(codes)
        print(workers_sessions)

        # Step 4: Determine roles based on sessions
        # Since both participants were able to connect, it means that there must be a valid session
        for session_idx, session in enumerate(workers_sessions[firstID]['expert_sessions']):
            # Session format -> dd/mm/yyyy_hh:mm-hh:mm
            initial_date_data = session.split('-')[0]
            initial_date = datetime.strptime(initial_date_data, '%d/%m/%Y_%H:%M')

            if current_date.day == initial_date.day and current_date.month == initial_date.month \
                    and current_date.year == initial_date.year and current_date.hour == initial_date.hour:
                # Here we have that the first participant is DE
                self.domain_expert_idx = 0
                self.proponent_idx = 1
                self.session_idx = session_idx
                return

        # If in-loop condition never occurs, then the first participant is the proponent
        # We still have to determine DE's session in order to retrieve the correct article
        # Here self.session_idx is None
        for session_idx, session in enumerate(workers_sessions[secondID]['expert_sessions']):
            # Session format -> dd/mm/yyyy_hh:mm-hh:mm
            initial_date_data = session.split('-')[0]
            initial_date = datetime.strptime(initial_date_data, '%d/%m/%Y_%H:%M')

            if current_date.day == initial_date.day and current_date.month == initial_date.month \
                    and current_date.year == initial_date.year and current_date.hour == initial_date.hour:
                # Here we have that the second participant is DE
                self.session_idx = session_idx
                return

    def retrieve_expert_paper(self):
        expert_code = self.workerIds[self.domain_expert_idx]

        try:
            expert_data = self.dataset[self.dataset['Authentication Code'] == expert_code]
            current_paper_data = expert_data.iloc[self.session_idx]
        except IndexError as e:
            print('Got -> session_idx={0},\n expert_data={1}'.format(self.session_idx, expert_data))
            current_paper_data = expert_data.iloc[expert_data.shape[0] - 1]

        return current_paper_data

    def parley(self):
        """
        For each agent, get an observation of the last action each of the other agents
        took.
        Then take an action yourself.
        """
        acts = self.acts
        if self.abstract_shown:
            self.current_turns += 1

        print("Turns -> ", self.current_turns)

        act_data = {}
        for act_idx, act in enumerate(acts):
            print("Act - {0}: {1}".format(act_idx, act))
            if act is not None:
                act_data['act_id'] = {key: value for key, value in act.items()}

        if act_data:
            self.conversation_data.append(act_data)

        # Step 1: DE has to read article content and P has to read article title
        # this part runs when both participants connect to the web application
        if not self.abstract_shown:

            # Determine roles
            self.determine_participants_roles()
            # self.dummy_determine_participant_roles()
            assert self.session_idx is not None  # Should always be True (useful for debugging)

            # Retrieve correct article for DE
            selected_article = self.retrieve_expert_paper()

            # Re-write agent names
            self.agents[self.proponent_idx].agent_id = "Proponent"
            self.agents[self.domain_expert_idx].agent_id = "Domain Expert"

            # Get context from dataloader
            # example_input = self.teacher.act()
            self.abstract = selected_article['Abstract']
            self.title = selected_article['Title']
            self.introduction = selected_article['Introduction']

            welcome_message = "Welcome! Loading role-specific interface. Take your time, you've got plenty!"

            acts[self.domain_expert_idx] = {"abstract": self.abstract,
                                            "title": self.title,
                                            "introduction": self.introduction,
                                            "text": welcome_message,
                                            "showFacts": True,
                                            "role": "Domain-Expert",
                                            "interaction": self.current_turns,
                                            "max_interactions": self.max_turns,
                                            'min_interactions': self.min_turns,
                                            "time": datetime.today().strftime('%d-%m-%Y-%H-%M-%S')
                                            }

            acts[self.proponent_idx] = {
                "title": self.title,
                "text": welcome_message,
                "showFacts": False,
                "role": "Proponent",
                "interaction": self.current_turns,
                "max_interactions": self.max_turns,
                'min_interactions': self.min_turns,
                "time": datetime.today().strftime('%d-%m-%Y-%H-%M-%S')
            }

            # get worker name from -> agent.mephisto_agent.get_worker().worker_name
            self.agents[self.domain_expert_idx].observe(validate(acts[self.domain_expert_idx]))
            self.agents[self.proponent_idx].observe(validate(acts[self.proponent_idx]))

            if not self.conversation_data:
                self.conversation_data.append({'abstract': self.abstract,
                                               'title': self.title,
                                               'introduction': self.introduction,
                                               'episode_done': False,
                                               'start_time': datetime.today().strftime('%d-%m-%Y-%H-%M-%S')})

            self.abstract_shown = True

            if self.time_limit_apply:
                self.time_limit_start = time.time()

            return

        # Random tip messages
        if self.current_tip_index < len(self.random_tip_points) and self.current_turns == self.random_tip_points[
            self.current_tip_index]:
            acts[self.domain_expert_idx] = {
                "text": self.tips[self.current_tip_index],
                "showFacts": True,
                "role": "Domain-Expert",
                "time": datetime.today().strftime('%d-%m-%Y-%H-%M-%S')
            }

            acts[self.proponent_idx] = {
                "text": self.tips[self.current_tip_index],
                "showFacts": False,
                "role": "Proponent",
                "time": datetime.today().strftime('%d-%m-%Y-%H-%M-%S')
            }

            self.agents[self.domain_expert_idx].observe(validate(acts[self.domain_expert_idx]))
            self.agents[self.proponent_idx].observe(validate(acts[self.proponent_idx]))

            # Update index
            self.current_tip_index += 1

        # Time messages
        if self.time_limit_apply:
            current_time = time.time()
            difference = current_time - self.time_limit_start

            if difference >= self.time_limit_first_notification and not self.time_limit_first_notification_sent:
                acts[self.domain_expert_idx] = {
                    "text": "15 minutes have passed! Please terminate the conversation in the next 5 minutes.",
                    "showFacts": True,
                    "role": "Domain-Expert",
                    "time": datetime.today().strftime('%d-%m-%Y-%H-%M-%S')
                }

                acts[self.proponent_idx] = {
                    "text": "15 minutes have passed! Please terminate the conversation in the next 5 minutes.",
                    "showFacts": False,
                    "role": "Proponent",
                    "time": datetime.today().strftime('%d-%m-%Y-%H-%M-%S')
                }

                self.agents[self.domain_expert_idx].observe(validate(acts[self.domain_expert_idx]))
                self.agents[self.proponent_idx].observe(validate(acts[self.proponent_idx]))

                self.time_limit_first_notification_sent = True

            if difference >= self.time_limit_second_notification and not self.time_limit_second_notification_sent:
                acts[self.domain_expert_idx] = {
                    "text": "You only have 1 minute to terminate the dialogue.",
                    "showFacts": True,
                    "role": "Domain-Expert",
                    "time": datetime.today().strftime('%d-%m-%Y-%H-%M-%S')
                }

                acts[self.proponent_idx] = {
                    "text": "You only have 1 minute to terminate the dialogue. "
                            "Make sure you check the 'Terminate Dialogue' checkbox before sending your last message.",
                    "showFacts": False,
                    "role": "Proponent",
                    "time": datetime.today().strftime('%d-%m-%Y-%H-%M-%S')
                }

                self.agents[self.domain_expert_idx].observe(validate(acts[self.domain_expert_idx]))
                self.agents[self.proponent_idx].observe(validate(acts[self.proponent_idx]))

                self.time_limit_second_notification_sent = True
                self.turn_timeout = 60

            if difference >= self.time_limit_end:
                self.episodeDone = True
                self.has_timed_out = True
                self.turn_timeout = 1   # force timeout
                print("[Parley] Ending episode due to timeout!")

            # Update turn timeout when close to time limit
            # if self.time_limit_end - current_time < self.turn_timeout:
            #     self.turn_timeout = self.time_limit_end - current_time

        for index in [self.proponent_idx, self.domain_expert_idx]:
            try:
                agent = self.agents[index]
                print("[Parley] Acting with timeout -> ", self.turn_timeout)
                acts[index] = agent.act(timeout=self.turn_timeout)
                if self.send_task_data:
                    acts[index].force_set(
                        "task_data",
                        {
                            "last_acting_agent": agent.agent_id,
                            "current_dialogue_turn": self.current_turns,
                            "utterance_count": self.current_turns + index,
                        },
                    )
            except TypeError:
                acts[index] = agent.act()  # not MTurkAgent

            # Add turn info
            acts[index].force_set(
                "interaction", self.current_turns,
            )

            acts[index].force_set(
                "max_interactions", self.max_turns,
            )

            acts[index].force_set(
                "min_interactions", self.min_turns,
            )

            acts[index].force_set(
                "time", datetime.today().strftime('%d-%m-%Y-%H-%M-%S'),
            )

            if 'episode_done' not in acts[index]:
                acts[index].force_set(
                    "episode_done", True
                )

            if acts[index]["episode_done"]:
                self.episodeDone = True
                self.conversation_data.append({
                    'end_time': datetime.today().strftime('%d-%m-%Y-%H-%M-%S')
                })

            for other_agent in self.agents:
                if other_agent != agent:
                    other_agent.observe(validate(acts[index]))
                    if other_agent == self.agents[self.domain_expert_idx] and self.episodeDone:
                        if self.has_timed_out:
                            other_msg = "Timeout has triggered chat end! The next message will be your last one!"
                        else:
                            other_msg = "Proponent has decided to terminate the conversation! " \
                                        "The next message will be your last one!"
                        other_agent.observe({
                            "text": other_msg
                        })

    def prep_save_data(self, agent):
        """Process and return any additional data from this world you may want to store"""
        print("A conversation has finished! Saving data...")
        return self.conversation_data

    def episode_done(self):
        return self.episodeDone

    def shutdown(self):
        """
        Shutdown all mturk agents in parallel, otherwise if one mturk agent is
        disconnected then it could prevent other mturk agents from completing.
        """
        global shutdown_agent

        def shutdown_agent(agent):
            try:
                agent.shutdown(timeout=None)
            except Exception:
                agent.shutdown()  # not MTurkAgent

        Parallel(n_jobs=len(self.agents), backend="threading")(
            delayed(shutdown_agent)(agent) for agent in self.agents
        )


def make_onboarding_world(opt, agent):
    return MultiAgentDialogOnboardWorld(opt, agent)


def retrieve_codes():
    df_path = os.path.join(cd.PROJECT_DIR, 'crowdsourcing', 'dataset.csv')
    df = pd.read_csv(df_path)

    codes = df['Authentication Code'].values
    return codes


def validate_onboarding(data):
    """Check the contents of the data to ensure they are valid"""
    # data['outputs']['messages'] -> list of messages
    print(f"Validating onboarding data {data}")

    # Get worker id
    worker_id = data['outputs']['messages'][0]['data']['state']['agent_display_name'].strip()

    # Get codes
    codes = retrieve_codes()

    valid = worker_id in codes

    if not valid:
        print("Worker {} is not valid! Gently kicking...".format(worker_id))

    return valid


def make_world(opt, agents):
    return MultiAgentDialogWorld(opt, agents)


def get_world_params():
    return {"agent_count": 2}
