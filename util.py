
import torch
import shutil
from os.path import join
import numpy as np
import random

def save_checkpoint(args, state, is_best, filename):
    model_path = join(args.output_folder, filename)
    torch.save(state, model_path)
    if is_best:
        shutil.copyfile(model_path, join(args.output_folder, "best_model.pth"))


def make_state(epoch_num, model, optimizer, recalls, best_r5, not_improved_num):
    return {
        "epoch_num": epoch_num,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "recalls": recalls,
        "best_r5": best_r5,
        "not_improved_num": not_improved_num,
        "random_state": random.getstate(),
        "numpy_random_state" : np.random.get_state(),
        "torch_state": torch.get_rng_state(),
        # TO-DO: check if needed
        "torch_random_state": torch.random.get_rng_state(),
        "torch_cuda_random_state": torch.cuda.get_rng_state()
    }


def recover_from_state(checkpoint_name, model, optimizer, resume_random=True):
    state = torch.load(checkpoint_name)
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
        torch.set_rng_state(state["torch_state"])
    return epoch_num, recalls, best_r5, not_improved_num