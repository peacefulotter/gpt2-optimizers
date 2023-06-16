import numpy as np
from torch.nn import Module
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer
from optimizers import Lion, Sophia, SignSGD
from transformers import (
    BertConfig,
    BertForMaskedLM, 
    T5Config,
    GPT2Config,
    GPT2LMHeadModel
    # T5ForMaskedLM
)

def get_bert_model(**kwargs):
    def cb(tokenizer):
        config = BertConfig(
            vocab_size=len(tokenizer),
            **kwargs
        )
        return BertForMaskedLM(config)
    return cb

def get_gpt2_model(**kwargs):
    def cb(tokenizer):
        config = GPT2Config(
            vocab_size=len(tokenizer),
            **kwargs
        )
        return GPT2LMHeadModel(config)
    return cb

def get_t5_model(**kwargs):
    def cb(tokenizer):
        config = T5Config(
            vocab_size=len(tokenizer),
            **kwargs
        )
        raise NotImplementedError()
        # return T5ForMaskedLM(config)
    return cb

def not_implemented():
    raise NotImplementedError()

def build_optimizer(Class: Optimizer):
    def cb(model: Module, **kwargs):
        return Class(params=model.parameters(), **kwargs)
    return cb

SEED = 42

OPTIMIZER_CONFIGS = {
    'adam': { 
        'build': build_optimizer(AdamW),
        'lrs': np.linspace(1e-4, 1e-3, 3),
        'default-lr':  1e-3     
    },
    'lion': { 
        'build': build_optimizer(Lion),
        'lrs': np.linspace(1e-5, 1e-4, 3),
        'default-lr':  1e-4  
    },
    'sophia': { 
        'build': build_optimizer(Sophia),
        'lrs': np.linspace(1e-5, 1e-4, 3),
        'default-lr':  1e-4  
    },
    'signsgd': { 
        'build': build_optimizer(SignSGD),
        'lrs': np.linspace(1e-4, 1e-3, 3),
        'default-lr':  1e-3 
    },
}

DATASET_CONFIGS = {
    'wikitext': {
        'dataset_path': 'wikitext',
        'dataset_name': 'wikitext-103-raw-v1'
    },
    'wikitext2':{
        'dataset_path': 'wikitext',
        'dataset_name': 'wikitext-2-raw-v1'
    }
}

MODEL_CONFIGS = {
    'mini-bert': {
        'max_seq_length': 512,
        'tokenizer_name': 'bert-base-cased',
        'model': get_bert_model(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=3072
        ),
        'mlm': True,
    },
    'bert': {
        'max_seq_length': 512,
        'tokenizer_name': 'bert-base-cased',
        'model': get_bert_model(),
        'mlm': True,
    },
    'gpt2': {
        'max_seq_length': 512,
        'tokenizer_name': 'gpt2',
        'model': get_gpt2_model(), # gpt2
        'mlm': False,
    },
    't5': {
        'max_seq_length': 512, # TODO: check max_seq_length
        'tokenizer_name': 't5-small',
        'model': get_t5_model, # t5
        'mlm': True
    }
}

