from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)
from peft import (
    get_peft_model,
    PeftModel,
    PeftConfig,
    LoraConfig,
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
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
from glue_sst2_utils import *
from example_glue_sst2_save_grad import *
from example_glue_sst2_preconditioner import *
from iEFOptimizer_GP import generalPurposeiEFOptimiser, cosine_similarity


def eval_preconditioner_vis(args, config):
    # load model
    base_model_name = config["model"]["base_model_name"]
    checkpoint_name = f"glue_sst2_{base_model_name}"
    print(f"Start training for glue-sst2 with {base_model_name} and seed {args.seed}")
    precond_eval_dir = args.expdir
    random_ckpt = False

    # switch initialisation based on peft_mode
    peft_mode = config["model"].get("peft_mode", "prompt")
    if peft_mode == "prompt":
        pass
    elif peft_mode == "lora":
        print("DOING LoRA")
        this_PE_num = 0
        if "model_ckpt" in config["model"] and config["model"]["model_ckpt"] != "none":
            print("[Eval] Eval Provided Checkpoint")
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
            print("[Eval] Eval Random Checkpoint")
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
            random_ckpt = True
            init_model_ckpt_path = os.path.join(
                args.expdir, checkpoint_name + "_init_seed{}".format(args.seed)
            )
            model.save_pretrained(init_model_ckpt_path)
            config["model"]["model_ckpt"] = init_model_ckpt_path

    model.print_trainable_parameters()

    # initialise exp directory
    precond_eval_dir = os.path.join(config["model"]["model_ckpt"], "precond-eval")
    if not os.path.exists(precond_eval_dir):
        os.makedirs(precond_eval_dir)
    # copy ckpt to this dir
    ckpt_base = os.path.basename(config["model"]["model_ckpt"])
    expdir_child = os.listdir(args.expdir)
    if ckpt_base not in expdir_child:
        os.symlink(config["model"]["model_ckpt"], os.path.join(args.expdir, ckpt_base))
    print("All results are saved in path:")
    print("************************************************")
    print(precond_eval_dir)
    print("************************************************")

    # prepare dataset
    if config["train_args"].get("remote", False):
        dataset = load_dataset(config["train_args"].get("data", "cifar100"))
    else:
        raise NotImplementedError

    print(dataset)

    image_processor = AutoImageProcessor.from_pretrained(base_model_name)

    normalize = Normalize(
        mean=image_processor.image_mean, std=image_processor.image_std
    )
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
        example_batch["pixel_values"] = [
            train_transforms(image.convert("RGB")) for image in example_batch["img"]
        ]
        return example_batch

    def preprocess_val(example_batch):
        """Apply val_transforms across a batch."""
        example_batch["pixel_values"] = [
            val_transforms(image.convert("RGB")) for image in example_batch["img"]
        ]
        return example_batch

    splits = dataset["train"].train_test_split(test_size=0.1)
    train_ds = splits["train"]
    val_ds = splits["test"]
    test_ds = dataset["test"]

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

    if not random_ckpt and config["train_args"].get(
        "require_sort", False
    ):  # order train dataloader if is not random_ckpt
        # load train sample idx order (according to target probability)
        train_order_idx_path = os.path.join(precond_eval_dir, "train_order_idx.npy")
        if os.path.exists(train_order_idx_path):
            train_order_idx = np.load(train_order_idx_path, allow_pickle=True)
            print("Successfully loaded the target probability ordering of samples")
        else:
            print(
                "====================================================================="
            )
            print("Start creating target probability ordering")
            target_prob_idx = []
            counter = 0
            for step, batch in tqdm(
                enumerate(train_dataloader), total=len(train_dataloader)
            ):
                batch = {k: v.to(args.device) for k, v in batch.items()}
                outputs = model(**batch)
                # recover target probability for every sample
                label_prob = get_target_prob_from_logits(
                    outputs.logits, batch["labels"]
                )
                # register target_prob
                for i in range(batch["labels"].shape[0]):
                    target_prob_idx.append((counter, label_prob[i]))
                    counter += 1
            target_prob_idx.sort(key=lambda x: x[1], reverse=True)
            train_order_idx = [i[0] for i in target_prob_idx]
            np.save(train_order_idx_path, train_order_idx, allow_pickle=True)
            print("Successfully generate the target probability ordering of samples")
        # reorder training samples
        train_order_idx = [int(idx) for idx in train_order_idx]
        random_train_dataloader = DataLoader(
            train_ds,
            shuffle=True,
            # collate_fn=default_data_collator,
            collate_fn=data_collator,
            batch_size=config["train_args"]["batch_size"],
            pin_memory=True,
        )
    else:
        random_train_dataloader = DataLoader(
            train_ds,
            shuffle=True,
            # collate_fn=default_data_collator,
            collate_fn=collate_fn,
            batch_size=config["train_args"]["batch_size"],
            pin_memory=True,
        )
        print("DO NOT use reordered train dataset for randomly initilised model")

    # register model with GPOpt
    GPOpt = generalPurposeiEFOptimiser(model)

    # do eval for the top 10000 target prob samples
    random_ray_list_path = os.path.join(precond_eval_dir, "random_rayleigh_list.npy")
    if os.path.exists(random_ray_list_path):
        print("Random Rayleigh ALREADY EVALUATED, which may be OVERWRITTEN")
    # start evalutation
    print("=====================================================================")
    print("Start Evaluating New Rayleighs")
    dampings = config["optimiser"]["damping"]
    if type(dampings) is list:
        for damping in dampings:
            random_ray_list_path_dmp = random_ray_list_path[:-4] + "_dmp-{}".format(damping) + ".npy"
            precond_evaluate(
                random_ray_list_path_dmp,
                config["train_args"]["eval_steps"],
                args.seed,
                model,
                GPOpt,
                random_train_dataloader,
                config["train_args"]["accumulate_steps"],
                args.device,
                damping=damping,
            )
    else:
        precond_evaluate(
            random_ray_list_path,
            config["train_args"]["eval_steps"],
            args.seed,
            model,
            GPOpt,
            random_train_dataloader,
            config["train_args"]["accumulate_steps"],
            args.device,
            damping=dampings,
        )