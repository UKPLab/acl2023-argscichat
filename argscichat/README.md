# Argumentative Models and Facts selection baselines #

In this project we provide:

- Experimental environment for training mentioned SciBERT argumentative models.

- Facts selection baselines: human lower-bound (3 annotators), tf-idf, [SentenceBert](https://www.sbert.net/) and their argumentative variants.

## Prerequisites

 - Install requirements as follows:

```
pip install -r requirements.txt
```


## Experiments

### Checking configuration files

To run an argumentative model, either on DrInventor or on ArgSciChat papers, you need to:

- Check your model configuration at `configs/model_config.json`. Make sure that your model configuration is correct.

- Set the correct data loader at `configs/data_loader.json`. Here's a quick summary map:

    - `dr_inventor_tokens`: for sequential argument component training on DrInventor.
    - `dr_inventor_components`: for argument link prediction training on DrInventor.
    - `arg_sci_chat_model_components_papers`: for annotating ArgSciChat detected arguments with link labels
    - `arg_sci_chat_tokens_papers`: for annotating ArgSciChat for argument component detection task
    
- Check `train_and_test_config.json` configuration file.

- Check `training_config.json` configuration file.

- Additionally, you might also want to check `callbacks.json` if you are using any available callback.

### Training an argumentative

Once your configuration files are all set, simply execute `runnables/test_train_and_test_v2.py`.

Your test results, including metrics, a copy of input configuration files and model weights, will be stored under `train_and_test` folder.

### Doing inference on a test corpus

If you want to test your argumentative model on an unlabelled dataset, like ArgSciChat, you have to:

- Check `configs/unseen_test_config.json` configuration file.

- Run `runnables/test_unseen_data.py`.

However, we have already set up an utility script for a pipeline annotation of ArgSciChat papers:

- The argument component sequential model is run on a token-level version of ArgSciChat paper corpus (check `local_database/papers_token_dataset_chunked.csv`(.

- The extracted arguments are then fed as input to the argument link prediction model (check `local_database/papers_model_components_dataset.csv`)

You can easily extract multiple versions of the corpus with different filtering threshold:

- Run `runnables/pipeline_annotate_papers.py`

The script can also be configured to compute statistics as shown in Figure 5 and Figure 7 in the paper.

Threshold-based extracted arguments with associated link labels are saved in `local_database/papers_model_components_dataset_*threshold*.csv`.

### Facts selection baselines

Depending on what kind of baselines you want to test, there are two scripts:

- `runnables/facts_selection_simple_baselines.py`: to compute standard tf-idf and SentenceBert baselines.

- `runnables/facts_selection_arg_simple_baselines.py`: to compute the argumentative version of tf-idf and SentenceBert baselines.

We also include, as in [QASPER](https://github.com/allenai/qasper-led-baseline), a first sentence and random baselines.
Due to their low performance, they were excluded in the paper.

*NOTE*: argumentative baselines require the existence of an argumentative ArgSciChat corpus corresponding to a given threshold value (check above section for more details).



