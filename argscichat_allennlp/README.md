# Allennlp LED baselines

In this project we provide:

- Training LED models for automatic DE reply generation.

## Prerequisites

 - Install requirements as follows:

```
pip install -r requirements.txt
```

## Experiments

We follow the same experimental setup as described in [QASPER](https://github.com/allenai/qasper-led-baseline).
Therefore, our code is a minor adapation of QASPER provided code to our case study.

We gratefully thank [QASPER](https://github.com/allenai/qasper-led-baseline) team for their work and code.

### LED baselines

All the configurations are under `training_config` folder. Data paths have to be set before running any experiment.

In order to run multiple training or prediction runs in a sequence, we have set up two simple utility scripts:

- `scripts/run_experiments.py`: modify the file settings section in order to define your own experiments. Models are saved under `model_runs` folder.

- `scripts/run_predictions.py`: runs model inference on saved models of your choice (check `model_runs` folder).

For what concerns model training, you can try different input combinations and model additional regularizations as follows:

```
{
    'folder_name': "your_run_name",
    'params': {
        'context': ["query", "article"],
        'pytorch_seed': 15371,
        'use_evidence_scaffold': False,
}
```

The `context` field accepts the following inputs: `[query, article, history, facts]`. For instance, if you want to run
the model with just the `query` as input, simply define `'context': ["query"]`.

Alike [QASPER](https://github.com/allenai/qasper-led-baseline), you can train your model with direct facts selection supervision 
by enabling `use_evidence_scaffold`. The code remains unchanged with the only exception that in QASPER there are multiple valid answers
for a given query.

### Reproducibility

If you want to reproduce our experiments, we have used the following seeds: `[15371, 15372, 15373]`.
Other seeds (e.g. numpy) are fixed and are equal to the ones employed in [QASPER](https://github.com/allenai/qasper-led-baseline).

### Facts selection baselines

Run `scripts/fact_selection_baselines.py` to compute baseline performance on all test folds.

## Dialogue Intents

We provide a version of ArgSciChat with dialogue intents in `argscichat_intents/dataset.csv`.