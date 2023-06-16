import torch
import tiktoken
import torch
import argparse
import config

from models.base import GPTBase


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config_format', default='base', choices=config.registered_formats())
    args, rem_args = parser.parse_known_args()
    return config.parse_args_with_format(format=args.config_format, base_parser=parser, args=rem_args, namespace=args)

def main(args):
    base_path = '/mlodata1/nmaire/optml-llm-baselines/llm-baselines/exps/wikitext/base'
    exp_path = 'h768_nh12_nlyr12_sl512_d005_adam_base_lr0.0001_bs50x4_1nodes_seed=0'

    checkpoint = torch.load(f'{base_path}/{exp_path}/ckpt.pt')
    # model = GPTBase(args).to('cuda:0')
    # model.load_state_dict(checkpoint['model'])
    tokenizer = tiktoken.get_encoding("gpt2")
    method_list = [method for method in dir(tokenizer) if not method.startswith('__')]
    print(method_list)
    sentence = 'Christophe Colomb was the first to '
    tokens = tokenizer.encode(sentence)
    print(tokens)

if __name__ == "__main__":
    args = get_args()
    main(args)
