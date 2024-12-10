import os
import yaml
import argparse
import random
import torch
import numpy as np
import wandb
from example_glue_sst2_train import *
from example_glue_sst2_train_vis import *
from example_glue_sst2_preconditioner import *
from example_glue_sst2_preconditioner_vis import eval_preconditioner_vis



parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=1337, type=int)
parser.add_argument("--device", default="cuda", help="model.to(device)")
parser.add_argument(
    "-m",
    "--mode",
    choices=["train", "train-vis", "evaluate-preconditioner", "evaluate-preconditioner-vis"],
    required=True,
)
parser.add_argument("-p", "--expdir", type=str, default=None)
parser.add_argument("--config", type=str, default=None)
parser.add_argument("--wandb_project", type=str, default=None)
parser.add_argument("--wandb_run_name", type=str, default=None)
args = parser.parse_args()


def main():
    if args.expdir is None:
        raise ValueError("No output directory found")
    elif args.mode in ("train", "train-vis", "evaluate-preconditioner", "evaluate-preconditioner-vis"):
        args.expdir = os.path.join(args.expdir, args.wandb_project, args.wandb_run_name)
        if not os.path.exists(args.expdir):
            os.makedirs(args.expdir)

    # load config, but if not provided, load from provided checkpoint, otherwise raise
    if args.config is not None:
        with open(args.config, "r") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    else:
        raise ValueError(
            "No config found. You have to provide either --config or --model_ckpt"
        )

    # Fix seed and make backends deterministic
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        with torch.cuda.device(args.device):
            torch.cuda.empty_cache()

    # cudnn enabled
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # preset data preprocessing arguments: instruction and prompt_init
    dataset_name = config["train_args"].get("data", "sst2")
    if dataset_name == "sst2":
        instruction =  "Classify if the following sentence's sentiment is positive or negative: "
        prompt_init = "Sentiment Classification Task: "
    elif dataset_name == "cola":
        instruction = "Classify if the following sentence's grammar is acceptable or unacceptable: "
        prompt_init = "Grammar Checking Task: "
    elif dataset_name == "qqp":
        instruction = "Classify if the following Q1 and Q2 are semantically equivalent, answer yes or no: "
        prompt_init = "Semantically Equivalent Checking Task: "
    elif dataset_name == "mnli" or dataset_name == "mnli_matched" or dataset_name == "mnli_mismatched":
        instruction = "Predict whether the premise entails the hypothesis, contradicts the hypothesis, or neither, answer yes, no or maybe: "
        prompt_init = "Textual Entailment Prediction Task: "
    elif dataset_name == "mrpc":
        instruction = "Classify if the following S1 and S2 are semantically equivalent, answer yes or no: "
        prompt_init = "Semantically Equivalent Checking Task: "
    elif dataset_name == "qnli":
        instruction = "Determine whether the context sentence S contains the answer to the question Q, answer yes or no: "
        prompt_init = "Textual Entailment Prediction Task: "
    elif dataset_name == "rte":
        instruction = "Classify if S1 entailment S2 or not, answer yes or no: "
        prompt_init = "Textual Entailment Recognition Task: "
    elif dataset_name == "stsb":
        instruction = "Determine the similarity score between S1 and S2, the score is from 0.0 to 5.0: "
        prompt_init = "Textual Similarity Scoring Task: "
    elif dataset_name == "wnli":
        instruction = "Determine if S2 with the pronoun substituted is entailed by S1: "
        prompt_init = "Textual Entailment Prediction Task: "
    else:
        instruction = None
        prompt_init = None
    config["model"]["instruction"] = instruction
    config["model"]["prompt_init"] = prompt_init

    if args.mode in ("train", "train-vis") and args.wandb_project is not None:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=config)

    if args.mode == "train":
        train_glue_sst2(args, config)
    elif args.mode == "train-vis":
        train_vis(args, config)
    elif args.mode == "evaluate-preconditioner":
        eval_preconditioner(args, config)
    elif args.mode == "evaluate-preconditioner-vis":
        eval_preconditioner_vis(args, config)


if __name__ == "__main__":
    main()
