from itertools import chain
from datasets import load_dataset
from transformers import AutoTokenizer
import sys
# Import configs
from configs import MODEL_CONFIGS, DATASET_CONFIGS

def get_training_corpus():
    return (
        raw_datasets["train"][i : i + 1000][text_column_name]
        for i in range(0, len(raw_datasets["train"]), 1000)
    )

# Main data processing function that will concatenate all texts from our dataset and generate chunks of
# max_seq_length.
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < max_seq_length  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // max_seq_length) * max_seq_length
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
        for k, t in concatenated_examples.items()
    }
    return result

if __name__ == '__main__':
    def _exit():
        print(f"""
            Usage: tokenizer.py <model> <dataset>
            Models: {MODEL_CONFIGS.keys()}
            Datasets: {DATASET_CONFIGS.keys()}
        """)
        sys.exit(1)

    if len(sys.argv) != 3:
        _exit()

    model_name = sys.argv[1]
    dataset_name = sys.argv[2]
    if model_name not in MODEL_CONFIGS.keys() or dataset_name not in DATASET_CONFIGS.keys():
        _exit()

    model_config = MODEL_CONFIGS[model_name]
    dataset_config = DATASET_CONFIGS[dataset_name]
    tokenizer_name = model_config['tokenizer_name']
    max_seq_length = model_config['max_seq_length']
    mlm = model_config['mlm']
    path = dataset_config['dataset_path']
    name = dataset_config['dataset_name']

    # Load dataset
    raw_datasets = load_dataset(path, name)
    column_names = list(raw_datasets["train"].features) # Evaluation: column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    # Use pretrained tokenizer and train it on new corpus
    training_corpus = get_training_corpus()
    old_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if not old_tokenizer.is_fast:
        raise Warning(f'Using pretrained {tokenizer_name} tokenizer is not a FAST tokenizer')
    tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, vocab_size=52000)

    def tokenize_function(examples):
        return tokenizer(
            examples[text_column_name], 
            truncation=True, 
            max_length=max_seq_length, 
            return_special_tokens_mask=mlm
        )


    # Tokenize dataset
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=8,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on every text in dataset",
    )

    # Group in batch the tokenized dataset
    tokenized_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=8,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {max_seq_length}",
    )

    tokenizer.save_pretrained(f'./save/{path}/{name}/tokenizer/{model_name}/') # tokenizer.json
    tokenized_datasets.save_to_disk(f'./save/{path}/{name}/datasets/{model_name}/')