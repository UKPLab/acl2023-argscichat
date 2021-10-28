local transformer_model = "allenai/led-base-16384";
local epochs = 50;
local patience = 20;
local batch_size = 1;
local num_gradient_accumulation_steps = 1;

local train_data_path = "/path/to/train.json";
local dev_data_path = "/path/to/val.json";

local training_data_size = 192;
local num_gpus = 1;

{
    "dataset_reader": {
        "type": "argscichat",
        "transformer_model_name": transformer_model,
        "max_document_length": 4000,
        "max_query_length": 1000,
        "include_argument_mask": false,
        "argument_mask_threshold": "0.7",
        "for_training": true,
        "context": ["query"]
    },
    "validation_dataset_reader": {
        "type": "argscichat",
        "transformer_model_name": transformer_model,
        "max_document_length": 4000,
        "max_query_length": 1000,
        "include_argument_mask": false,
        "argument_mask_threshold": "0.7",
        "for_training": false,
        "context": ["query"]
    },
    "train_data_path": train_data_path,
    "validation_data_path": dev_data_path,
    "vocabulary": {
        "type": "empty",
    },
    "model": {
        "type": "argscichat_baseline",
        "transformer_model_name": transformer_model,
        "attention_window_size": 700,
        "gradient_checkpointing": true,
        "use_evidence_scaffold": false,
        "include_argument_mask": false,
        "attention_dropout": 0.1,
    },
    "data_loader": {
        "batch_size": batch_size,
    },
    "trainer": {
      "optimizer": {
        "type": "adam",
        "lr": 5e-5,
      },
      "learning_rate_scheduler": {
        "type": "slanted_triangular",
        "num_epochs": epochs,
        "cut_frac": 0.1,
        "num_steps_per_epoch": std.ceil(training_data_size / (batch_size * num_gradient_accumulation_steps * num_gpus)),
      },
      "grad_clipping": 1.0,
      "num_epochs": epochs,
      "num_gradient_accumulation_steps": num_gradient_accumulation_steps,
      "patience": patience,
      "validation_metric": "+evidence_f1",
      "enable_default_callbacks": false,
      "use_amp": true
    },
    "pytorch_seed": 15372,
}
