# gpt2-optimizers
### CS-439 Optimization for Machine Learning

#### Code structure
- huggingface: Attempt to run GPT2 using huggingface packages, did not work in the end
  - Quickstart: first train the tokenizer and use it to tokenize the dataset `python ./huggingface/tokenizer.py {model} {dataset}`. Then train the model `python ./huggingface/train.py {model} {dataset} {optimizer}`
- llm-baselines: Code for the "Exploring Optimizers on Causal Language Models" report
  - `src/config/` Arguments for training scripts
  - `src/data/` Functions to download and tokenize famous datasets such as openwebtext, wikitext, shakespeare, and more
  - `src/models/` Models and layers classes
  - `src/optim/` Training loop and optimizers
  - `src/main.py` Training file that parses the arguments, downloads the dataset, instantiate the model and starts training 


#### Scripts
Scripts to run the experiments can be found under `llm-baselines/scripts/`. Many parameters are available, a list can be found under `./llm-baselines/src/config/base.py`.


For more infos, see `llm-baselines/README.md`
