from transformers import AutoImageProcessor, AutoModelForImageClassification
from peft import (
    get_peft_model,
    LoraConfig,
    PeftModel,
    PeftConfig,
)
try:
    from torchvision.transforms import (
        CenterCrop,
        Compose,
        Normalize,
        RandomHorizontalFlip,
        RandomResizedCrop,
        Resize,
        ToTensor,
    )
except:
    print("no torchvision!")
import torch
from datasets import load_dataset
import os
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from tqdm import tqdm
import random
import numpy as np
import wandb
from transformers import Adafactor
from torch.optim import AdamW, SGD
from iEFOptimizer import *
from iEFOptimizer_GP import generalPurposeiEFOptimiser
from glue_sst2_utils import *
from example_glue_sst2_save_grad import *


def train_vis(args, config):
    # load model
    base_model_name = config["model"]["base_model_name"]
    checkpoint_name = f"glue_sst2_{base_model_name.split('/')[-1]}"
    print(f"Start training for glue-sst2 with {base_model_name} and seed {args.seed}")

    # switch initialisation based on peft_mode
    peft_mode = config["model"].get("peft_mode", "prompt")
    if peft_mode == "prompt":
        pass # TODO
    elif peft_mode == "lora":
        print("DOING LoRA")
        if "model_ckpt" in config["model"] and config["model"]["model_ckpt"] != "none":
            print("[Train] Train from Provided Checkpoint")
            # load model
            peft_config = PeftConfig.from_pretrained(config["model"]["model_ckpt"])
            model = AutoModelForImageClassification.from_pretrained(
                peft_config.base_model_name_or_path
            )
            model = PeftModel.from_pretrained(
                model, config["model"]["model_ckpt"], is_trainable=True
            )
            model = model.to(args.device)
        else:
            print("[Train] Start from Scratch")
            peft_config = LoraConfig(
                r=config["model"]["LR_num"],
                lora_alpha=config["model"]["LR_alpha"],
                target_modules=["query", "value"],
                lora_dropout=config["model"]["LR_dropout"],
                bias=config["model"]["LR_bias"],
            )
            model = AutoModelForImageClassification.from_pretrained(base_model_name)
            model = get_peft_model(model, peft_config)
            model = model.to(args.device)

    model.print_trainable_parameters()

    # prepare dataset
    if config["train_args"].get("remote", False):
        dataset_name = config["train_args"].get("data", "mnist")
        dataset = load_dataset(dataset_name)
    else:
        raise NotImplementedError
    
    print(dataset)

    image_processor = AutoImageProcessor.from_pretrained(base_model_name)

    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    train_transforms = Compose(
        [
            RandomResizedCrop(image_processor.size["height"]),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

    val_transforms = Compose(
        [
            Resize(image_processor.size["height"]),
            CenterCrop(image_processor.size["height"]),
            ToTensor(),
            normalize,
        ]
    )


    def preprocess_train(example_batch):
        """Apply train_transforms across a batch."""
        example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["img"]]
        return example_batch


    def preprocess_val(example_batch):
        """Apply val_transforms across a batch."""
        example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["img"]]
        return example_batch

    splits = dataset['train'].train_test_split(test_size=0.1)
    train_ds = splits['train']
    val_ds = splits['test']
    test_ds = dataset['test']

    train_ds.set_transform(preprocess_train)
    val_ds.set_transform(preprocess_val)
    test_ds.set_transform(preprocess_val)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["coarse_label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    train_dataloader = DataLoader(
        train_ds,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=config["train_args"]["batch_size"],
        pin_memory=True,
    )
    eval_dataloader = DataLoader(
        val_ds,
        collate_fn=collate_fn,
        batch_size=config["train_args"]["eval_batch_size"],
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_ds,
        collate_fn=collate_fn,
        batch_size=config["train_args"]["eval_batch_size"],
        pin_memory=True,
    )

    # optimizer and lr scheduler
    opt_type = config["optimiser"]["opt_type"]
    if "const_weight" in config["optimiser"]:
        const_weight = config["optimiser"]["const_weight"]
    else:
        const_weight = False
    if "norm_weight" in config["optimiser"]:
        norm_weight = config["optimiser"]["norm_weight"]
    else:
        norm_weight = False
    if "sort_with" in config["optimiser"]:
        sort_with = config["optimiser"]["sort_with"]
    else:
        sort_with = "grad_norm"
    if "max_samples" in config["optimiser"]:
        max_samples = config["optimiser"]["max_samples"]
    else:
        max_samples = None
    if "mean_truncate" in config["optimiser"]:
        mean_truncate = config["optimiser"]["mean_truncate"]
    else:
        mean_truncate = None
    if "adapt_lr" in config["optimiser"]:
        adapt_lr = config["optimiser"]["adapt_lr"]
    else:
        adapt_lr = False
    if "converged_prob_thresh" in config["optimiser"]:
        converged_prob_thresh = config["optimiser"]["converged_prob_thresh"]
    else:
        converged_prob_thresh = None
    if "momentum2" in config["optimiser"]:
        momentum2 = config["optimiser"]["momentum2"]
    else:
        momentum2 = 0.999
    if "momentum" in config["optimiser"]:
        momentum = config["optimiser"]["momentum"]
    else:
        momentum = 0.9
    if "cossim_bound" in config["optimiser"]:
        cossim_bound = config["optimiser"]["cossim_bound"]
    else:
        cossim_bound = None
    if "adafactor" in opt_type.lower():
        optimizer = Adafactor(
            model.parameters(),
            lr=config["optimiser"]["lr"],
            weight_decay=config["optimiser"]["af_weight_decay"],
            scale_parameter=config["optimiser"]["af_scale_parameter"],
            relative_step=config["optimiser"]["af_relative_step"],
            warmup_init=config["optimiser"]["af_warmup_init"],
        )
    elif "adamw" in opt_type.lower():
        optimizer = AdamW(
            model.parameters(),
            lr=config["optimiser"]["lr"],
            weight_decay=config["optimiser"]["af_weight_decay"],
            betas=[momentum, momentum2],
        )
    elif "torchsgd" in opt_type.lower():
        optimizer = SGD(
            model.parameters(),
            lr=config["optimiser"]["lr"],
            weight_decay=config["optimiser"]["af_weight_decay"],
        )
    elif opt_type.startswith("GP"): # general purpose iEF optimiser is indicated
        opt_type = opt_type[2:]
        optimizer = generalPurposeiEFOptimiser(
            model,
            lr=config["optimiser"]["lr"],
            clip_norm=config["optimiser"].get("clip_norm", None),
            opt_flag=opt_type,
            grad_renorm=False,
            damping=config["optimiser"]["damping"],
            momentum=config["optimiser"]["momentum"],
            momentum2=momentum2,
            norm_update=config["optimiser"]["norm_update"],
            const_weight=const_weight,
            norm_weight=norm_weight,
            max_samples=max_samples,
            mean_truncate=mean_truncate,
            adapt_lr=adapt_lr,
            sort_with=sort_with,
            converged_prob_thresh=converged_prob_thresh,
            cossim_bound=cossim_bound,
            line_search=config["optimiser"].get("line_search", False),
            line_search_beta=config["optimiser"].get("line_search_beta", 0.1),
            line_search_alpha=config["optimiser"].get("line_search_alpha", 0.001),
        )

    if "scheduler" in config and config["scheduler"]["constant"]:
        lr_scheduler = ExponentialLR(
            optimizer=optimizer,
            gamma=1
        )
    elif "scheduler" in config and config["scheduler"]["plateau"]:
        lr_scheduler = ReduceLROnPlateau(
            optimizer=optimizer,
            factor=config["scheduler"]["factor"],
            patience=config["scheduler"]["patience"],
        )
    else:
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=int(
                len(train_dataloader)
                * config["train_args"]["total_epochs"]
                / config["train_args"]["accumulate_steps"]
            ),
        )

    # training and evaluation
    global_steps = 0
    accumulate_cnt = 0
    accumulate_batches = []
    tmp_train_loss = 0
    tmp_train_total_cnt = 0
    tmp_train_correct_cnt = 0
    best_eval_acc = -1
    for epoch in tqdm(range(config["train_args"]["total_epochs"])):
        # train for this epoch
        model.train()
        train_epoch_loss = 0
        train_epoch_total_cnt = 0
        train_epoch_correct_cnt = 0
        for step, batch in tqdm(
            enumerate(train_dataloader), total=len(train_dataloader)
        ):
            # select model mode during training
            if (
                "eval_train" in config["train_args"]
                and config["train_args"]["eval_train"]
            ):
                model.eval()
            else:
                model.train()

            # do update to model
            accumulate_batches.append(batch)
            batch = {k: v.to(args.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            outputs.logits.retain_grad()
            loss.backward(retain_graph=True)
            accumulate_cnt += 1

            # for sampled fisher optimisation method, do fisher sampling
            if opt_type.lower().startswith("sf"):
                MC_TIME = int(opt_type[2:])
                optimizer.fisher_sampling(outputs.logits, batch["labels"], MC_TIME=MC_TIME, retain_graph=False)

            # register error signal at logits level
            if "adafactor" not in opt_type.lower() and "adamw" not in opt_type.lower() and "torchsgd" not in opt_type.lower():
                optimizer.update_logits_grad(
                    outputs.logits.grad, loss, outputs.logits, batch["labels"]
                )

            # collect stats
            train_epoch_loss += loss.detach().float()
            target_prob, correct_cnt, total_cnt = evaluator_vis(
                outputs.logits,
                batch["labels"],
            )
            train_epoch_total_cnt += total_cnt
            train_epoch_correct_cnt += correct_cnt
            tmp_train_loss += loss.detach().float()
            tmp_train_total_cnt += total_cnt
            tmp_train_correct_cnt += correct_cnt

            # update model when batch cnt is reached
            if accumulate_cnt == config["train_args"]["accumulate_steps"]:
                if "adafactor" in opt_type.lower() or "adamw" in opt_type.lower() or "torchsgd" in opt_type.lower():
                    optimizer.step()
                else: # iEF optimiser require a closure to enable line-search
                    # define a closure that evaluates on a given batch of data
                    def closure():
                        model.eval()
                        total_loss = 0
                        with torch.no_grad():
                            for batch in accumulate_batches:
                                batch = {k: v.to(model.device) for k, v in batch.items()}
                                outputs = model(**batch)
                                total_loss += outputs.loss.item()
                        return total_loss
                    optimizer.step(closure, batch_loss=tmp_train_loss)
                # schedule update size
                if "scheduler" not in config or "plateau" not in config["scheduler"] or not config["scheduler"]["plateau"]:
                    lr_scheduler.step()
                optimizer.zero_grad()
                # report train stats
                wandb.log(
                    {
                        f"train.loss": tmp_train_loss / accumulate_cnt,
                        f"train.acc": tmp_train_correct_cnt / tmp_train_total_cnt * 100,
                    },
                    step=global_steps,
                )
                if global_steps >= 1:
                    if (
                        hasattr(optimizer, "stats")
                        and "original_grad_norm" in optimizer.stats
                    ):
                        wandb.log(
                            {
                                f"stat/Original Gradient Norm": optimizer.stats[
                                    "original_grad_norm"
                                ],
                                f"stat/Improved Gradient Norm": optimizer.stats[
                                    "new_grad_norm"
                                ],
                                f"stat/Update Norm": optimizer.stats["update_norm"],
                                f"stat/Original Gradients Similarity": optimizer.stats[
                                    "original_grad_sim"
                                ],
                                f"stat/Improved Gradients Similarity": optimizer.stats[
                                    "new_grad_sim"
                                ],
                                f"stat/Updates Similarity": optimizer.stats[
                                    "update_sim"
                                ],
                                f"adv stat/Gradient Similarity within Batch": optimizer.stats[
                                    "batch_grad_sim"
                                ],
                                f"adv stat/Improved Gradient Effectiveness": optimizer.stats[
                                    "new_grad_effectivenss"
                                ],
                                f"adv stat/Sample Gradients Norms Segregation": optimizer.stats[
                                    "sample_grad_10%max/min"
                                ],
                                f"adv stat/Logits Gradients Norms Segregation": optimizer.stats[
                                    "logits_grad_10%max/min"
                                ],
                                f"adv stat/Gradients Norms Segregation (std.mean)": optimizer.stats[
                                    "sample_grad_std/mean"
                                ],
                                f"adv stat/Sample Gradients Max Norm": optimizer.stats[
                                    "sample_grad_max"
                                ],
                                f"prob stat/Label Probability Segretation": optimizer.stats[
                                    "prob_10%max/min"
                                ],
                                f"prob stat/Label Probability Segretation (std.mean)": optimizer.stats[
                                    "prob_std/mean"
                                ],
                                f"prob stat/Label Probability Average": optimizer.stats[
                                    "prob_mean"
                                ],
                                f"prob stat/Converged Data Ratio": optimizer.stats[
                                    "converged_count"
                                ]
                                / (
                                    config["train_args"]["batch_size"]
                                    * config["train_args"]["accumulate_steps"]
                                ),
                                f"weight stat/Weight Norm": optimizer.stats[
                                    "weight_norm"
                                ],
                                f"precision stat/iEF rel error": optimizer.stats[
                                    "rel_err"
                                ],
                                f"sample stat/iEF sample cnt": optimizer.stats[
                                    "max_samples"
                                ],
                            },
                            step=global_steps,
                        )
                # increment step and reset stats
                global_steps += 1
                tmp_train_loss = 0
                tmp_train_total_cnt = 0
                tmp_train_correct_cnt = 0
                accumulate_cnt = 0
                accumulate_batches = []

                optimizer.disable_persample_grad()

                # evaluate when required
                if global_steps % config["train_args"]["eval_steps"] == 0:
                    ret, ret_probs = eval_model_vis(model, args.device, eval_dataloader)
                    wandb.log(ret, step=global_steps)
                    if ret["eval.acc"] > best_eval_acc:
                        best_eval_acc = ret["eval.acc"]
                        best_acc_ret = ret
                        # save model
                        model.save_pretrained(
                            os.path.join(args.expdir, checkpoint_name)
                        )
                    if "scheduler" in config and "plateau" in config["scheduler"] and config["scheduler"]["plateau"]:
                        lr_scheduler.step(ret["eval.loss"])

                optimizer.enable_persample_grad()

            # clear cuda cache if necessary
            torch.cuda.empty_cache()

        # record train stats
        train_epoch_loss = train_epoch_loss / len(train_dataloader)
        train_epoch_acc = train_epoch_correct_cnt / train_epoch_total_cnt
        wandb.log(
            {
                f"train-epoch.loss": train_epoch_loss,
                f"train-epoch.acc": train_epoch_acc * 100,
            },
            step=epoch,
        )

        # record eval stats again at the end of each epoch
        optimizer.disable_persample_grad()
        ret, ret_probs = eval_model_vis(model, args.device, eval_dataloader)
        optimizer.enable_persample_grad()
        wandb.log(
            ret,
            step=global_steps,
        )
        if ret["eval.acc"] > best_eval_acc:
            best_eval_acc = ret["eval.acc"]
            best_acc_ret = ret
            # save model
            model.save_pretrained(
                os.path.join(args.expdir, checkpoint_name)
            )
        model.save_pretrained(os.path.join(args.expdir, "{}_e{}".format(checkpoint_name, epoch)))
        
        # set random seed using epoch id
        random.seed(args.seed + epoch + 1)
        np.random.seed(args.seed + epoch + 1)
        torch.manual_seed(args.seed + epoch + 1)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + epoch + 1)
            with torch.cuda.device(args.device):
                torch.cuda.empty_cache()
    
    # report best validation result
    best_ret_reformat = {}
    for k, v in best_acc_ret.items():
        best_ret_reformat["End / acc-best-" + k] = v
    wandb.log(best_ret_reformat)

    # update checkpoint name
    final_checkpoint_name = checkpoint_name + "_eval-acc:{:.3f}".format(best_acc_ret["eval.acc"])
    os.rename(
        os.path.join(args.expdir, checkpoint_name),
        os.path.join(args.expdir, final_checkpoint_name),
    )

    peft4eval = PeftConfig.from_pretrained(
        os.path.join(args.expdir, final_checkpoint_name)
    )
    model4eval = AutoModelForImageClassification.from_pretrained(
        peft4eval.base_model_name_or_path
    )
    model4eval = PeftModel.from_pretrained(
        model4eval,
        os.path.join(args.expdir, final_checkpoint_name)
    )
    model4eval = model4eval.to(args.device)

    eval_ret, _ = eval_model_vis(model4eval, args.device, test_dataloader)
    print("Test Acc. {}".format(eval_ret["eval.acc"]))