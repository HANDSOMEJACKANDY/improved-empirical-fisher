from transformers import AutoModelForSeq2SeqLM
from peft import (
    get_peft_model,
    PromptTuningInit,
    PromptTuningConfig,
    TaskType,
    PeftModel,
    PeftConfig,
    LoraConfig,
)
import torch
from datasets import load_dataset
import os
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import random
import numpy as np
from subprocess import call
from glue_sst2_utils import *
from example_glue_sst2_save_grad import *
from iEFOptimizer_GP import generalPurposeiEFOptimiser, cosine_similarity
import copy


def eval_preconditioner(args, config):
    # load model
    base_model_name = config["model"]["base_model_name"]
    task_name = config["train_args"]["data"]
    checkpoint_name = f"glue_{task_name}_{base_model_name}"
    print(f"Start training for glue-sst2 with {base_model_name} and seed {args.seed}")
    precond_eval_dir = args.expdir
    random_ckpt = False

    # modify adapter_config.json for remote!!!
    if config["train_args"].get("remote", False):
        pass

    model_ckpts = config["model"].get("model_ckpt", "none")
    config_lists = []
    if type(model_ckpts) == str:
        config_lists = [config]
    elif type(model_ckpts) is list:
        for ckpt in model_ckpts:
            this_config = copy.deepcopy(config)
            this_config["model"]["model_ckpt"] = ckpt
            config_lists.append(this_config)
    
    for config in config_lists:
        # switch initialisation based on peft_mode
        peft_mode = config["model"].get("peft_mode", "prompt")
        if peft_mode == "prompt":
            print("DOING PROMPT TUNING")
            if (
                "model_ckpt" not in config["model"]
                or config["model"]["model_ckpt"] == "none"
            ):
                print("[Eval] Eval Random Ckpt")
                if type(config["model"]["prompt_init"]) is str:
                    peft_config = PromptTuningConfig(
                        task_type=TaskType.CAUSAL_LM,
                        prompt_tuning_init=PromptTuningInit.TEXT,
                        num_virtual_tokens=config["model"]["PE_num"],
                        prompt_tuning_init_text=config["model"]["prompt_init"],
                        tokenizer_name_or_path=base_model_name,
                    )
                else:
                    peft_config = PromptTuningConfig(
                        task_type=TaskType.CAUSAL_LM,
                        prompt_tuning_init=PromptTuningInit.RANDOM,
                        num_virtual_tokens=config["model"]["PE_num"],
                        tokenizer_name_or_path=base_model_name,
                    )
                model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
                model = get_peft_model(model, peft_config)
                # update checkpoint name
                init_model_ckpt_path = os.path.join(
                    args.expdir, checkpoint_name + "_init_seed{}".format(args.seed)
                )
                model.save_pretrained(init_model_ckpt_path)
                config["model"]["model_ckpt"] = init_model_ckpt_path
                this_PE_num = config["model"]["PE_num"]
                random_ckpt = True
            if "model_ckpt" in config["model"] and config["model"]["model_ckpt"] != "none":
                # load ckeckpoint prompt-tuning model
                # complicated loading method to enable backprop to soft-prompts (weird PEFT module design)
                print("[Eval] Eval Provided Checkpoint")
                # load model
                loaded_peft_config = PeftConfig.from_pretrained(
                    config["model"]["model_ckpt"]
                )
                loaded_model = AutoModelForSeq2SeqLM.from_pretrained(
                    loaded_peft_config.base_model_name_or_path
                )
                loaded_model = PeftModel.from_pretrained(
                    loaded_model, config["model"]["model_ckpt"]
                )
                E = list(loaded_model.prompt_encoder["default"].parameters())[0].to(
                    args.device
                )
                # copy pretrained parameter to untrained model (by pass cannot-finetune-PT issue)
                peft_config = PromptTuningConfig(
                    task_type=TaskType.CAUSAL_LM,
                    prompt_tuning_init=PromptTuningInit.RANDOM,
                    num_virtual_tokens=E.shape[0],
                    tokenizer_name_or_path=base_model_name,
                )
                model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
                model = get_peft_model(model, peft_config)
                model = model.to(args.device)
                E_new = list(model.prompt_encoder["default"].parameters())[0]
                E_new.data.copy_(E.data)
                this_PE_num = E.shape[0]
            PE = model.prompt_encoder["default"]
        elif peft_mode == "lora":
            print("DOING LoRA")
            this_PE_num = 0
            if "model_ckpt" in config["model"] and config["model"]["model_ckpt"] != "none":
                print("[Eval] Eval Provided Checkpoint")
                # load model
                peft_config = PeftConfig.from_pretrained(config["model"]["model_ckpt"])
                model = AutoModelForSeq2SeqLM.from_pretrained(
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
                    target_modules=["q", "v"],
                    lora_dropout=config["model"]["LR_dropout"],
                    bias=config["model"]["LR_bias"],
                )
                model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
                model = get_peft_model(model, peft_config)
                model = model.to(args.device)
                # update checkpoint name
                init_model_ckpt_path = os.path.join(
                    args.expdir, checkpoint_name + "_init_seed{}".format(args.seed)
                )
                model.save_pretrained(init_model_ckpt_path)
                config["model"]["model_ckpt"] = init_model_ckpt_path
                random_ckpt = True

        model.print_trainable_parameters()

        # clearify number of prompts
        if (
            peft_mode == "prompt"
            and not random_ckpt
            and config["model"]["PE_num"] != E.shape[0]
        ):
            print(
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )
            print(
                "Mismatch between specified PE_num: {} and ckpt's PE_num: {}!!!".format(
                    config["model"]["PE_num"], E.shape[0]
                )
            )
            print()
            config["model"]["PE_num"] = E.shape[0]

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
            dataset_name = ("glue", config["train_args"].get("data", "sst2"))
            dataset = load_dataset(*dataset_name)
        else:
            raise NotImplementedError

        if config["train_args"].get("data", "sst2") == "qqp" or config["train_args"].get("data", "sst2") == "mrpc":
            classes = ["no", "yes"]
            dataset = dataset.map(
                lambda x: {"text_label": [classes[label] for label in x["label"]]},
                batched=True,
                num_proc=1,
            )
        elif config["train_args"].get("data", "sst2") == "rte" or config["train_args"].get("data", "sst2") == "qnli":
            classes = ["yes", "no"]
            dataset = dataset.map(
                lambda x: {"text_label": [classes[label] for label in x["label"]]},
                batched=True,
                num_proc=1,
            )
        elif config["train_args"].get("data", "sst2") == "mnli":
            classes = ["yes", "maybe", "no"]
            dataset = dataset.map(
                lambda x: {"text_label": [classes[label] for label in x["label"]]},
                batched=True,
                num_proc=1,
            )
        else:
            classes = [k.replace("_", " ") for k in dataset["train"].features["label"].names]
            dataset = dataset.map(
                lambda x: {"text_label": [classes[label] for label in x["label"]]},
                batched=True,
                num_proc=1,
            )

        if config["train_args"].get("data", "sst2") == "qqp":
            dataset = dataset.map(
                lambda x: {"sentence": ["Q1: {} Q2: {}".format(q1, q2) for q1, q2 in zip(x["question1"], x["question2"])]},
                batched=True,
                num_proc=1,
            )
        elif config["train_args"].get("data", "sst2") == "mnli":
            dataset = dataset.map(
                lambda x: {"sentence": ["premise: {} hypothesis: {}".format(p, h) for p, h in zip(x["premise"], x["hypothesis"])]},
                batched=True,
                num_proc=1,
            )
        elif config["train_args"].get("data", "sst2") == "mrpc" or config["train_args"].get("data", "sst2") == "rte":
            dataset = dataset.map(
                lambda x: {"sentence": ["S1: {} S2: {}".format(s1, s2) for s1, s2 in zip(x["sentence1"], x["sentence2"])]},
                batched=True,
                num_proc=1,
            )
        elif config["train_args"].get("data", "sst2") == "qnli":
            dataset = dataset.map(
                lambda x: {"sentence": ["Q: {} S: {}".format(q, s) for q, s in zip(x["question"], x["sentence"])]},
                batched=True,
                num_proc=1,
            )
        print(dataset)

        # data preprocessing
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["base_model_name"])
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        processed_datasets = dataset.map(
            preprocess_function,
            fn_kwargs={
                "tokenizer": tokenizer,
                "text_column": "sentence",
                "label_column": "text_label",
                "max_length": config["train_args"]["max_length"],
                "instruction": config["model"]["instruction"],
            },
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )

        # do inference on all training samples to get target_prob for every sample for reordering of training set
        train_dataloader = DataLoader(
            processed_datasets["train"],
            shuffle=False,
            # collate_fn=default_data_collator,
            collate_fn=data_collator,
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
                processed_datasets["train"],
                shuffle=True,
                # collate_fn=default_data_collator,
                collate_fn=data_collator,
                batch_size=config["train_args"]["batch_size"],
                pin_memory=True,
            )
        else:
            random_train_dataloader = DataLoader(
                processed_datasets["train"],
                shuffle=True,
                # collate_fn=default_data_collator,
                collate_fn=data_collator,
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

def update_generation(
    Js, logits_grads, target_probs, J_hat, damping=1e-7, filter_duplicate_thresh=0.95
):
    """
    Compute the iEF, iEF(lg), EF, SGD and Adam gradient from a list of Jacobian matrix Js. In Js, every J's first dimension is assumed to be the batch dimension.
    """
    # stack all Js
    # J_org = torch.cat(Js, dim=0)
    J_org = Js
    J = J_org.view(J_org.shape[0], -1)

    # compute logit_grad norm
    J_lg_org = torch.cat(logits_grads, dim=0)
    Jl = J_lg_org.view(J_lg_org.shape[0], -1)
    grad_logits_norms = Jl.norm(dim=1).to(torch.float64)

    # compute the covariance of J
    grad_inputs_norms = J.norm(dim=1).to(torch.float64)
    C = J @ J.T
    C = C.to(torch.float64)

    # remove highly similar (duplicate) directions
    if 0 < filter_duplicate_thresh and filter_duplicate_thresh <= 1:
        """
        One example of removing highly duplicate samples

        A gradient with similarity -1.00000 is removed
        Label Prob: 0.90
        Rayleighs Rel: {'SGD': 1.0, 'EF': 2.8456233365377788, 'iEF': 0.531804715519998, 'iEFlg': 0.36505905433991426}
        Effectiveness: {'SGD': 1.0, 'EF': 0.004461147933508774, 'iEF': 0.21630646072018514, 'iEFlg': 0.5568310717653663}
        Max / Min Param Grad Norm ratio:  8332.7
        Max / Min Logits Grad Norm ratio: 3258.7

        The high similarity gradient is not removed
        Label Prob: 0.90
        Rayleighs Rel: {'SGD': 1.0, 'EF': 2.668001535276764, 'iEF': 0.5203463599768995, 'iEFlg': 7.236976914807328}
        Effectiveness: {'SGD': 1.0, 'EF': 0.004570350198310135, 'iEF': 0.23001792996448497, 'iEFlg': 0.0018401063737793512}
        Max / Min Param Grad Norm ratio:  8332.7
        Max / Min Logits Grad Norm ratio: 3258.7

        the duplicate direction significantly impaired the quality of almost undamped iEF update.
        """
        # compute grad similarity
        diagonal_vector = C.diag() ** 0.5
        cov_norm = diagonal_vector.unsqueeze(dim=-1) @ diagonal_vector.unsqueeze(dim=0)
        cov_sim = C / cov_norm

        # for highly similar gradients, preserve only the larger norm one
        remove_idx = []
        J_row_cnt = J.shape[0]
        for i in range(J_row_cnt):
            for j in range(i + 1, J_row_cnt):
                if torch.abs(cov_sim[i, j]) > filter_duplicate_thresh:
                    print(
                        "A gradient with similarity {:.5f} is removed".format(
                            cov_sim[i, j]
                        )
                    )
                    if grad_inputs_norms[j] > grad_inputs_norms[i]:
                        remove_idx.append(i)
                    else:
                        remove_idx.append(j)
        grad_idx = [i for i in range(J_row_cnt) if i not in remove_idx]
        if len(grad_idx) != J_row_cnt:
            # retain the parameter vector when const_weight is activated
            J = J[grad_idx, :]
            C = C[grad_idx, :][:, grad_idx]
            cov_sim = cov_sim[grad_idx, :][:, grad_idx]
            grad_inputs_norms = grad_inputs_norms[grad_idx]
            grad_logits_norms = grad_logits_norms[grad_idx]

    # compute SGD grad
    SGD_grad = J.sum(dim=0).view(J_org.shape[1:])

    # compute damping factor (using sampled Fisher as reference)
    J_hat = J_hat.to(torch.float64)
    J_hat, C_hat_o = remove_J_C_duplicates(J_hat, thresh=0.99, retain_C_thresh=1)
    if damping < 0:
        damping = C_hat_o.diag().max() * -damping

    # Compute SF Update
    g = SGD_grad.to(torch.float64)
    I_hat = torch.eye(C_hat_o.shape[0], dtype=C_hat_o.dtype, device=C_hat_o.device)
    C_hat = C_hat_o + damping * I_hat
    J_hat_g = J_hat @ g
    C_hat_i_J_hat_g = torch.linalg.solve(C_hat, J_hat_g)
    J_hat_T_C_hat_i_J_hat_g = J_hat.T @ C_hat_i_J_hat_g
    SF_grad = 1 / damping * (g - J_hat_T_C_hat_i_J_hat_g)
    iEF_grad = SF_grad

    # add damping to covariance
    # scaled damping
    damping_matrix = torch.eye(
        C.shape[0], dtype=C.dtype, device=C.device
    )  # torch.diag(C.diag())
    damped_C = C + damping * damping_matrix

    # Ci = torch.inverse((1 - damping) * C + damping * damping_matrix)
    # ignore overly small eigen-values
    L, Q = torch.linalg.eigh(damped_C)

    # L[L <= ratio_thresh] = 1 # ignore overly small eigenvalues
    # L[:10] = 1
    Ci = Q @ torch.diag(L**-1) @ Q.T

    target_grad_weighting = (Ci @ (grad_logits_norms**2)).to(torch.float32)
    grad_flat = target_grad_weighting @ J
    iEFlg_grad = grad_flat.view(J_org.shape[1:])

    # compute EF grad
    target_grad_weighting = (Ci @ torch.ones_like(grad_inputs_norms)).to(torch.float32)
    # target_grad_weighting = torch.linalg.solve(damped_C, torch.ones_like(grad_inputs_norms)).to(torch.float32)
    grad_flat = target_grad_weighting @ J
    EF_grad = grad_flat.view(J_org.shape[1:])

    # output_grads
    return (
        (
            SGD_grad,
            EF_grad,
            iEF_grad,
            iEFlg_grad,
        ),
        ("SGD", "EF", "iEF", "iEFlg"),
        grad_inputs_norms.cpu(),
        grad_logits_norms.cpu(),
        damping,
    )


def precond_evaluate(
    result_path,
    eval_steps,
    seed,
    model,
    GPOpt,
    dataloader,
    accumulate_steps,
    device,
    damping=1e-7,
):
    # initialise
    Rayleigh_list = []
    model.eval()
    # get Prompt length
    epoch = -1
    while True:
        epoch += 1
        # do never inherite gradients accross epochs. This could lead to singular covariance matrix
        accumulate_cnt = 0
        accumulate_batches = []
        label_probs = []
        logits_grads = []
        output_logits = []
        GPOpt.clear_saved_grad()
        for step, cpu_batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            # record involved batches
            accumulate_batches.append(cpu_batch)

            # do inference and get gradient
            batch = {k: v.to(device) for k, v in cpu_batch.items()}
            if len(batch["labels"].shape) == 2:
                batch["labels"] = batch["labels"][
                    :, 0:1
                ]  # only care about the first label token (eos token training is not considered)
            outputs = model(**batch)
            outputs.logits.retain_grad()
            outputs.loss.backward(retain_graph=True)
            logits_grads.append(outputs.logits.grad.detach())
            output_logits.append(outputs.logits.detach().cpu())
            accumulate_cnt += 1

            # accumulate fisher sampling's jacobian
            MC_TIME = 1
            GPOpt.fisher_sampling(
                outputs.logits, batch["labels"], MC_TIME=MC_TIME, retain_graph=False
            )

            # get target probability for debug
            label_prob = get_target_prob_from_logits(outputs.logits, batch["labels"])
            label_probs.append(label_prob)

            # DO precond evaluation when one batch is ready
            if accumulate_cnt == accumulate_steps:
                # get standard Jacobian first
                J = GPOpt.compute_Jacobian(SF=False)

                # get sampled Jacobian
                J_hat = GPOpt.collect_SF_Jacobian()

                # compute SGD, iEF and EF updates
                updates, grad_names, grad_norm, logits_grad_norm, final_damping = (
                    update_generation(
                        J, logits_grads, label_probs, damping=damping, J_hat=J_hat
                    )
                )
                # convert flat updates to correct shape updates
                updates = [GPOpt.flat_grad_to_grad_list(u) for u in updates]

                # clear cache
                # pt_module.saved_grad_inputs = []
                logits_grads = []
                output_logits = []
                J = None
                J_hat = None
                torch.cuda.empty_cache()

                # compute the vGv for various updates
                GPOpt.disable_persample_grad()
                vGv_sum = None
                for batch in accumulate_batches:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    if len(batch["labels"].shape) == 2:
                        batch["labels"] = batch["labels"][:, 0:1]
                    outputs = model(**batch)
                    outputs.logits.retain_grad()
                    outputs.loss.backward(retain_graph=True)
                    GPOpt.clear_saved_grad()
                    # compute vGvs
                    vGvs = vGvp_multi(
                        outputs.loss,
                        outputs.logits,
                        GPOpt.trainable_parameters_sequence,
                        updates,
                    )
                    if vGv_sum is None:
                        vGv_sum = np.array(vGvs)
                    else:
                        vGv_sum += np.array(vGvs)
                    # release cache
                    outputs.loss = None
                    outputs.logits = None
                    outputs = None
                    torch.cuda.empty_cache()
                GPOpt.enable_persample_grad()

                # compute Rayleigh Ratios
                Rayleighs, vgs = vHv_vg_multi_icml(vGv_sum, updates, updates[0])
                Rayleighs_dict = {}
                Rayleighs_dict["Rayleighs_rel"] = {
                    gn: ray_rel / Rayleighs[0]
                    for gn, ray_rel in zip(grad_names, Rayleighs)
                }
                Rayleighs_dict["Rayleighs"] = {
                    gn: ray_rel for gn, ray_rel in zip(grad_names, Rayleighs)
                }
                Rayleighs_dict["vGv"] = {
                    gn: vGv_s for gn, vGv_s in zip(grad_names, vGv_sum)
                }
                Rayleighs_dict["vg"] = {gn: vg for gn, vg in zip(grad_names, vgs)}
                Rayleighs_dict["Effectiveness"] = {
                    gn: cosine_similarity(update, updates[0])[0]
                    for gn, update in zip(grad_names, updates)
                }
                Rayleighs_dict["Update_Norm"] = {
                    gn: GPOpt.grad_list_to_flat_grad(update).norm().item()
                    for gn, update in zip(grad_names, updates)
                }
                Rayleighs_dict["target_probabilities"] = torch.cat(label_probs).numpy()
                Rayleighs_dict["logits_grad_norm"] = logits_grad_norm.numpy()
                Rayleighs_dict["grad_norm"] = grad_norm.numpy()
                Rayleighs_dict["final_damping"] = final_damping
                print(
                    "Label Prob: {:.2f}".format(
                        Rayleighs_dict["target_probabilities"].mean()
                    )
                )
                print("Rayleighs Rel:", Rayleighs_dict["Rayleighs_rel"])
                print("Effectiveness:", Rayleighs_dict["Effectiveness"])
                print("Update Norm:", Rayleighs_dict["Update_Norm"])
                print(
                    "Final Damping:                    {}".format(
                        Rayleighs_dict["final_damping"]
                    )
                )
                print(
                    "Max / Min Param Grad Norm ratio:  {:.1f}".format(
                        grad_norm.max() / grad_norm.min()
                    )
                )
                print(
                    "Max / Min Logits Grad Norm ratio: {:.1f}".format(
                        logits_grad_norm.max() / logits_grad_norm.min()
                    )
                )
                Rayleigh_list.append(Rayleighs_dict)

                # reset accumulate stats
                accumulate_cnt = 0
                GPOpt.clear_saved_grad()
                label_probs = []
                accumulate_batches = []

                # when enough evaluation is computed, stop evaluation
                if len(Rayleigh_list) == eval_steps:
                    np.save(result_path, Rayleigh_list, allow_pickle=True)
                    return

        # set random seed using epoch id
        random.seed(seed + epoch + 1)
        np.random.seed(seed + epoch + 1)
        torch.manual_seed(seed + epoch + 1)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + epoch + 1)
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
