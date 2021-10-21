# Data Collection Tool

In this project, we provide:

- Mephisto data collection interface

- Dialogues extraction from Mephisto database

All provided scripts are a simple extension of the Mephisto [bootstrap chat example](https://github.com/facebookresearch/Mephisto/tree/main/packages/bootstrap-chat).

We strongly encourage you to have a look at [Mephisto docs](https://github.com/facebookresearch/Mephisto) before testing our data collection tool.

## Prerequisites

- Download and install [Mephisto](https://github.com/facebookresearch/Mephisto).

- Download and install [Parlai](https://github.com/facebookresearch/ParlAI).

- [Optional] Download and install [Heroku](https://devcenter.heroku.com/articles/heroku-cli) (if you want to run the data collection tool non-locally)

**NOTE**: If you are working with an IDE, like Pycharm, and you have installed Mephisto or Parlai via the github repo, you might need to add the source folders to your project.

## Running the tool

The core components of a Mephisto-based application are `worlds.py` and `run.py` scripts.

- `worlds.py`: defines the business logic of the application. In our case, the alternated interaction between two dialogue partners

- `run.py`: represents the main script that launches the whole Mephisto-based application. 
In particular, it is possible to configure the application behaviour by modifying its internal configuration:

```
defaults = [
    {"mephisto/blueprint": BLUEPRINT_TYPE},
    {"mephisto/architect": "heroku"},
    {"mephisto/provider": "mock"},
    {"conf": "sci_chat"},
]
```

### Configuration files

Before launching the data collection tool, please check the following configuration files:

- **Mephisto app config**: each configuration file is a .yaml file stored in `conf/` folder. It defines the general information of your Mephisto application.

- **Personal config**: contains information about the internal database you might have setup for your web interface (see below section for more info).
This information is used in the `worlds.py/read_from_db()` function. This function retrieves user information, such as dialogue sessions and authentication code.
Such retrieved data is used to validate the user and to retrieve its dialogue session data: role determination, paper content retrieval, etc..

In our experiments, [our data collection website](http://arg-sci-chat.ukp.informatik.tu-darmstadt.de/) was associated to an internal Mysql database.
The database was used to stored participants information, such as authentication code, DE session slots, P session slots, paper content, etc..

This information is then exploited by the Mephisto data collection application to validate if users are connecting to the right dialogue session and retrieve corresponding information in case of success.

### Execution

In this example, we are running our application on `heroku`, with a `mock` provider and by using `sci_chat` configuration file (check `conf/` folder).
For instance, if you want to run the application locally, you can change `mephisto/architect` to `local` instead of `heroku`.
For other possibilities, we recommend you to check the official [Mephisto](https://github.com/facebookresearch/Mephisto) documentation.

Once, everything is set up:

- Run your application by running `run.py`.

- Your application should be up and running and reachable at the provided Url.

## Extracting and saving dialogues

Run the `examine_results.py` script by specifying the task name from which you want to extract dialogues (check `task_name` field within your Mephisto config file).

In particular, the script saves two distinct .csv files:

- **Summary file**: each row denotes one single dialogue information along with general statistics: assignment, average duration, number of interactions, etc...

- **Dialogue file**: it stores dialogue information, distinguishing by dialogue roles. It also contains metadata (assignment, unit ID) to match summary file information.