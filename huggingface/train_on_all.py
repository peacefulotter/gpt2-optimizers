import sys
from train import train
from configs import MODEL_CONFIGS, DATASET_CONFIGS, OPTIMIZER_CONFIGS  

"""
Method to train on all configs listed in configs.py
"""
def train_on_all(model_name, dataset_name):
    for optim_name, optimizer in OPTIMIZER_CONFIGS.items():
        for lr in optimizer['lrs']:
            print('\n\n====================================')
            print(f'\tTraining on {optim_name} - {lr}')
            print('====================================')
            train(model_name, dataset_name, optim_name, lr=lr)


if __name__ == "__main__":
    def _exit():
        print(f"""
            Usage: train.py <model> <dataset> \n
            Models: {MODEL_CONFIGS.keys()}\n
            Datasets: {DATASET_CONFIGS.keys()}\n
        """)
        sys.exit(1)

    l = len(sys.argv)
    if l < 3:
        _exit()

    model_name = sys.argv[1]
    dataset_name = sys.argv[2]
    if (
        model_name not in MODEL_CONFIGS.keys() or 
        dataset_name not in DATASET_CONFIGS.keys()
    ):
        _exit()

    train_on_all(model_name, dataset_name)
       

    