
import torch
import shutil
import os.path
import numpy as np
import random
import logging
import sos_loss

def save_checkpoint(args, state, is_best, filename):
    model_path = os.path.join(args.output_folder, filename)
    torch.save(state, model_path)
    if is_best:
        shutil.copyfile(model_path, os.path.join(args.output_folder, "best_model.pth"))


def make_state(args, epoch_num, model, optimizer, scheduler, recalls, best_r5, not_improved_num):
    return {
        "args": args,
        "epoch_num": epoch_num,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "recalls": recalls,
        "best_r5": best_r5,
        "not_improved_num": not_improved_num,
        "random_state": random.getstate(),
        "numpy_random_state" : np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "torch_cuda_random_state": torch.cuda.get_rng_state()   # TO-DO: check if needed
    }


def load_state(state_path):
    if not os.path.isfile(state_path): raise FileNotFoundError(f"Checkpoint '{state_path}' does not exist")
    return torch.load(state_path)


def load_args_from_state(state, parsed_args):
    args = state["args"]
    args.exp_name = parsed_args.exp_name
    args.datasets_folder = parsed_args.datasets_folder
    args.epochs_num = parsed_args.epochs_num
    args.patience = parsed_args.patience
    args.resume = parsed_args.resume
    args.test_only = parsed_args.test_only
    return args


def resume_from_state(state, model, optimizer, scheduler, restore_random=True):
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None: scheduler.load_state_dict(state["scheduler_state_dict"])
    epoch_num = state["epoch_num"]
    recalls = state["recalls"]
    best_r5 = state["best_r5"]
    not_improved_num = state["not_improved_num"]
    # For total reproducibility, restore the random generators states
    if restore_random:
        random.setstate(state["random_state"])
        np.random.set_state(state["numpy_random_state"])
        torch.set_rng_state(state["torch_random_state"])
        torch.cuda.set_rng_state(state["torch_cuda_random_state"])  # TO-DO: check if needed
    return epoch_num, recalls, best_r5, not_improved_num


def get_optimizer(args, model):
    if args.optim == "sgd":
        momentum = 0.9
        weight_decay = 0.001
        scheduler_step_size = 5
        scheduler_gamma = 0.1
        logging.debug(f"Using SGD optimizer (lr: {args.lr}, momentum: {momentum}, weight-decay {weight_decay})")
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    elif args.optim == "adam":
        logging.debug(f"Using Adam optimizer (lr: {args.lr})")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = None
    else:
        raise RuntimeError(f"Unknown optimizer {args.optim}")
    return optimizer, scheduler

def get_loss(args):
    if args.loss == "triplet":
        logging.debug(f"Using Torch's Triplet Loss (margin: {args.margin})")
        criterion = torch.nn.TripletMarginLoss(margin=args.margin, p=2, reduction="sum")
    if args.loss == "sare_joint" or args.loss== "sare_ind":
        logging.debug(f"Using {args.loss} Loss")
        criterion = None
        #to use SARE we need to calculate features before, we do that in train.py
    if args.sos:
        logging.debug("Using SOS loss")
        criterion_sos = sos_loss.SOSLoss()
    else:
        criterion_sos = None
    return criterion, criterion_sos