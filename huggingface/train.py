import json
import math
import wandb
import torch
import gc, sys
import numpy as np
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from datasets import load_from_disk
from transformers import (
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer, 
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from configs import (
    SEED, 
    MODEL_CONFIGS, 
    DATASET_CONFIGS,
    OPTIMIZER_CONFIGS
)
from datetime import datetime as dt
import evaluate


class MetricsCallback(TrainerCallback):
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        loss = kwargs['metrics']['eval_loss']
        pp = math.exp(loss)
        wandb.log({'eval/perplexity': pp}, commit=False)

class LogCallback(TrainerCallback):
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        logs = state.log_history
        if 'loss' in logs[-1].keys():
            loss = logs[-1]['loss']
            pp = math.exp(loss)
            wandb.log({'train/perplexity': pp}, commit=False)

accuracy_metric = evaluate.load("accuracy")
exact_match_metric = evaluate.load("exact_match")
mean_iou = evaluate.load("mean_iou")
perplexity = evaluate.load("perplexity", module_type="metric")

def compute_metrics(pred):    
    logits = torch.from_numpy(pred.predictions)
    predictions = np.argmax(logits, axis=-1)
    references = torch.from_numpy(pred.label_ids)
    results = accuracy_metric.compute(predictions=predictions, references=references)
    print(results)
    results = exact_match_metric.compute(predictions=predictions, references=references)
    print(results)
    results = mean_iou.compute(predictions=predictions, references=references, num_labels=10, ignore_index=-100)
    print(results)
    results = perplexity.compute(predictions=predictions, model_id='gpt2')
    print(results)

def train(model_name, dataset_name, optimizer_name, lr=None):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    model_config = MODEL_CONFIGS[model_name]
    dataset_config = DATASET_CONFIGS[dataset_name]
    optimizer_config = OPTIMIZER_CONFIGS[optimizer_name]
    # tokenizer_name = model_config['tokenizer_name']
    path = dataset_config['dataset_path']
    name = dataset_config['dataset_name']

    with open(f'./save/{path}/{name}/tokenizer/{model_name}/special_tokens_map.json') as f:
        special_tokens = json.load(f)
    
    tokenized_datasets = load_from_disk(f'./save/{path}/{name}/datasets/{model_name}/')
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=f'./save/{path}/{name}/tokenizer/{model_name}/tokenizer.json',
        verbose=False,
        **special_tokens
    )
    
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    mlm = model_config['mlm']
    model = model_config['model'](tokenizer)
    model.to(device)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=mlm)
    lr = lr if lr is not None else optimizer_config['default-lr']
    optimizer: Optimizer = optimizer_config['build'](model, lr=lr)
    time_now = dt.now().strftime("%m/%d/%Y, %H:%M:%S")

    training_args = TrainingArguments(
        output_dir=f'./save/{model_name}/output/',
        logging_dir=f'./save/{model_name}/logs/',
        evaluation_strategy = 'steps',
        gradient_accumulation_steps=4,
        eval_accumulation_steps=4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        seed=SEED,
        bf16=True,
        bf16_full_eval=True,
        eval_steps=1,
        run_name=f"{model_name}-{dataset_name}-{optimizer_name}-{lr}-{time_now}",
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),
        callbacks=[MetricsCallback, LogCallback],
    )

    torch.cuda.empty_cache()
    gc.collect()

    trainer.evaluate()
    trainer.train()
    trainer.save_model(f"./save/{model_name}/output/{optimizer.__class__.__name__}")

    eval_results = trainer.evaluate()
    print(f"{optimizer.__class__.__name__} {optimizer.defaults['lr']} - results: {eval_results}")


if __name__ == "__main__":
    def _exit():
        print(f"""
            Usage: train.py <model> <dataset> <optimizer>
            Models: {MODEL_CONFIGS.keys()}
            Datasets: {DATASET_CONFIGS.keys()}
            Optimizers: {OPTIMIZER_CONFIGS.keys()}
            lr: float > 0
        """)
        sys.exit(1)

    l = len(sys.argv)
    if l < 4:
        _exit()

    model_name = sys.argv[1]
    dataset_name = sys.argv[2]
    optimizer_name = sys.argv[3]
    lr = sys.argv[4] if l >= 5 else None
    if (
        model_name not in MODEL_CONFIGS.keys() or 
        dataset_name not in DATASET_CONFIGS.keys() or 
        optimizer_name not in OPTIMIZER_CONFIGS.keys()
    ):
        _exit()

    train(model_name, dataset_name, optimizer_name, lr)
       

    