# !/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import collection_tool.const_define as cd
import os
from mephisto.operations.operator import Operator
from mephisto.tools.scripts import load_db_and_process_config
from mephisto.abstractions.blueprints.parlai_chat.parlai_chat_blueprint import (
    BLUEPRINT_TYPE,
    SharedParlAITaskState,
)
from mephisto.operations.hydra_config import RunScriptConfig, register_script_config

import hydra
from omegaconf import DictConfig
from dataclasses import dataclass, field
from typing import List, Any
import pandas as pd
from collection_tool.utility.json_utils import load_json
from getpass import getpass

PROJECT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
TASK_DIRECTORY = os.path.join(cd.LOCAL_DATASETS_DIR, 'argscichat_collection')

defaults = [
    {"mephisto/blueprint": BLUEPRINT_TYPE},
    {"mephisto/architect": "heroku"},
    {"mephisto/provider": "mock"},
    {"conf": "sci_chat"},
]


@dataclass
class ScriptConfig(RunScriptConfig):
    defaults: List[Any] = field(default_factory=lambda: defaults)
    project_dir: str = PROJECT_DIRECTORY
    task_dir: str = TASK_DIRECTORY
    num_turns: int = field(
        default=15,
        metadata={"help": "Number of turns before a conversation is complete"},
    )

    min_num_turns: int = field(
        default=7,
        metadata={"help": "Minimum number of turns required"},
    )

    event_id: int = field(
        default=1,
        metadata={"help": "Unique identifier of the data collection process"}
    )

    turn_timeout: int = field(
        default=240,
        metadata={
            "help": "Maximum response time before kicking "
                    "a worker out, default 240 seconds (4 minutes)"
        },
    )

    time_limit: int = field(
        default=1200,
        metadata={
            "help": "Time limit for discussion (20 minutes)"
        }
    )

    first_notification: int = field(
        default=900,
        metadata={
            "help": "First notification regarding dialogue time limit (15 minutes)"
        }
    )

    second_notification: int = field(
        default=1140,
        metadata={
            "help": "Second notification regarding dialogue time limit (19 minutes)"
        }
    )

    monitoring_log_rate: int = field(
        default=10,
        metadata={
            'help': 'Frequency in seconds of logging the monitoring of the crowdsourcing task'
        },
    )

    dataset_path: str = field(
        default=os.path.join(PROJECT_DIRECTORY, 'dataset.csv'),
        metadata={
            "help": "Path of .csv file containing article data for each author"
        }
    )

    # teacher: TeacherConfig = TeacherConfig()


register_script_config(name="scriptconfig", module=ScriptConfig)


@hydra.main(config_name="scriptconfig")
def main(cfg: DictConfig) -> None:
    # this reads the config .yaml file
    db, cfg = load_db_and_process_config(cfg)

    collection_config = load_json(os.path.join(cd.CONFIGS_DIR, cd.JSON_COLLECTION_CONFIG_NAME))

    # teacher = get_teacher(cfg)
    world_opt = {
        "num_turns": cfg.num_turns,
        "min_num_turns": cfg.min_num_turns,
        "turn_timeout": cfg.turn_timeout,
        "time_limit": cfg.time_limit,
        'first_notification': cfg.first_notification,
        'second_notification': cfg.second_notification,
        "dataset": pd.read_csv(cfg.dataset_path),
        'event_id': cfg.event_id,
        'db_admin': collection_config['collection_db_admin'],
        'db_password': getpass('Input your internal db password'),
        'db_host': collection_config['collection_db_host'],
        'db_name': collection_config['collection_db_name']
    }

    custom_bundle_path = cfg.mephisto.blueprint.get("custom_source_bundle", None)
    if custom_bundle_path is not None:
        assert os.path.exists(custom_bundle_path), (
            "Must build the custom bundle with `npm install; npm run dev` from within "
            f"the {PROJECT_DIRECTORY}/webapp directory in order to demo a custom bundle "
        )
        world_opt["send_task_data"] = True

    shared_state = SharedParlAITaskState(
        world_opt=world_opt, onboarding_world_opt=world_opt
    )

    operator = Operator(db)

    operator.validate_and_run_config(cfg.mephisto, shared_state)
    operator.wait_for_runs_then_shutdown(skip_input=True, log_rate=cfg.monitoring_log_rate)


if __name__ == "__main__":
    main()
