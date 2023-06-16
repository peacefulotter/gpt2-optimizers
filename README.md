# optml-project

#### Code structure
- huggingface: Attempt to run GPT2 using huggingface packages, did not work in the end
  - Quickstart: `python ./huggingface/train.py {model} {dataset} {optimizer}`
- llm-baselines: Code for the "Exploring Optimizers on Causal Language Models" report
  - `src/config/` Arguments for training scripts
  - `src/data/` Functions to download and tokenize famous datasets such as openwebtext, wikitext, shakespeare, and more
  - `src/models/` Models and layers classes
  - `src/optim/` Training loop and optimizers
  - `src/main.py` Training file that parses the arguments, downloads the dataset, instantiate the model and starts training 


For more infos, see `llm-baselines/README.md`