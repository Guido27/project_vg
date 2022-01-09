
import torch
import shutil
import os.path
import numpy as np
import random

def save_checkpoint(args, state, is_best, filename):
    model_path = os.path.join(args.output_folder, filename)
    torch.save(state, model_path)
    if is_best:
        shutil.copyfile(model_path, os.path.join(args.output_folder, "best_model.pth"))


def make_state(args, epoch_num, model, optimizer, recalls, best_r5, not_improved_num):
    return {
        "args": args,
        "epoch_num": epoch_num,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "recalls": recalls,
        "best_r5": best_r5,
        "not_improved_num": not_improved_num,
        "random_state": random.getstate(),
        "numpy_random_state" : np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "torch_cuda_random_state": torch.cuda.get_rng_state()   # TO-DO: check if needed
    }


def load_state(state_path):
    if os.path.isfile(state_path):
        return torch.load(state_path)
    return None


def load_args_from_state(state, args):
    '''
    Resume the ars from the state, preserving:
    exp_name, datasets_folder and resume_model params
    '''
    exp_name = args.exp_name
    datasets_folder = args.datasets_folder
    resume_model = args.resume_model
    args = state["args"]
    args.exp_name = exp_name
    args.datasets_folder = datasets_folder
    args.resume_model = resume_model


def resume_from_state(state, model, optimizer, resume_random=True):
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    epoch_num = state["epoch_num"]
    recalls = state["recalls"]
    best_r5 = state["best_r5"]
    not_improved_num = state["not_improved_num"]
    # For total reproducibility, restore the random generators states
    if resume_random:
        random.setstate(state["random_state"])
        np.random.set_state(state["numpy_random_state"])
        torch.set_rng_state(state["torch_random_state"])
        torch.cuda.set_rng_state(state["torch_cuda_random_state"])  # TO-DO: check if needed
    return epoch_num, recalls, best_r5, not_improved_num