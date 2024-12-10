from collections.abc import Mapping
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from tqdm import tqdm


def preprocess_function(
    examples, tokenizer, text_column, label_column, max_length, instruction=""
):
    batch_size = len(examples[text_column])
    inputs = [instruction + str(x) for x in examples[text_column]]
    targets = [str(x) for x in examples[label_column]]
    model_inputs = tokenizer(inputs)
    labels = tokenizer(targets)
    this_max_length = 0
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = sample_input_ids
        labels["input_ids"][i] = label_input_ids
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
        if len(sample_input_ids) > this_max_length:
            this_max_length = len(sample_input_ids)
    # find max length of batch
    # max_length = min(max_length, this_max_length)
    # align to max length
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            max_length - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (
            max_length - len(sample_input_ids)
        ) + model_inputs["attention_mask"][i]
        labels["input_ids"][i] = label_input_ids
        model_inputs["input_ids"][i] = torch.tensor(
            model_inputs["input_ids"][i][:max_length]
        )
        model_inputs["attention_mask"][i] = torch.tensor(
            model_inputs["attention_mask"][i][:max_length]
        )
        labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# evaluator
def evaluator(logits, labels, tokenizer, PE_num=10, task="sst2"):
    # get target probability for get given samples
    target_prob = get_target_prob_from_logits(logits, labels)
    # evaluation
    greedy_pred = torch.argmax(logits, dim=-1)
    decoded_output = tokenizer.batch_decode(greedy_pred[:, PE_num:], skip_special_tokens=True)
    labels[labels == -100] = tokenizer.pad_token_id
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    if task == "cola":
        correct_cnt = 0
        total_cnt = 0
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for l, o in zip(decoded_labels, decoded_output):
            total_cnt += 1
            o = o.lower()
            if len(o) == 0:
                continue
            if o == l:
                correct_cnt += 1

            if l == "acceptable":
                if o == "acceptable":
                    tp += 1
                else:
                    fn += 1
            elif l == "unacceptable":
                if o == "unacceptable":
                    tn += 1
                else:
                    fp += 1
        return target_prob, correct_cnt, total_cnt, tp, tn, fp, fn
    elif task == "qqp" or task == "mrpc":
        correct_cnt = 0
        total_cnt = 0
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for l, o in zip(decoded_labels, decoded_output):
            total_cnt += 1
            o = o.lower()
            if len(o) == 0:
                continue
            if o == l:
                correct_cnt += 1

            if l == "yes":
                if o == "yes":
                    tp += 1
                else:
                    fn += 1
            elif l == "no":
                if o == "no":
                    tn += 1
                else:
                    fp += 1
        return target_prob, correct_cnt, total_cnt, tp, tn, fp, fn
    else:
        correct_cnt = 0
        total_cnt = 0
        for l, o in zip(decoded_labels, decoded_output):
            total_cnt += 1
            o = o.lower()
            if len(o) == 0:
                continue
            if o == l:
                correct_cnt += 1
        return target_prob, correct_cnt, total_cnt


def eval_model(model, device, eval_dataloader, tokenizer, PE_num=10, task="sst2"):
    # evaluate once per log steps
    model.eval()
    eval_loss = 0
    eval_total_cnt = 0
    eval_correct_cnt = 0
    eval_tp = 0
    eval_tn = 0
    eval_fp = 0
    eval_fn = 0
    target_probability = []
    for step, batch in tqdm(enumerate(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        eval_loss += loss.detach().float()

        if task == "cola" or task == "qqp" or task == "mrpc":
            target_prob, correct_cnt, total_cnt, tp, tn, fp, fn = evaluator(outputs.logits, batch["labels"], tokenizer, PE_num=PE_num, task=task)
            eval_total_cnt += total_cnt
            eval_correct_cnt += correct_cnt
            eval_tp += tp
            eval_tn += tn
            eval_fp += fp
            eval_fn += fn
        else:
            target_prob, correct_cnt, total_cnt = evaluator(outputs.logits, batch["labels"], tokenizer, PE_num=PE_num, task=task)
            eval_total_cnt += total_cnt
            eval_correct_cnt += correct_cnt
        
        # collect target_probability for later analysis
        target_probability.append(target_prob)

    # compute loss and metric results
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)
    eval_epoch_acc = eval_correct_cnt / eval_total_cnt
    if task == "cola":
        eval_epoch_metric2 = (eval_tp * eval_tn - eval_fp * eval_fn) / np.sqrt((eval_tp + eval_fp) * (eval_tp + eval_fn) * (eval_tn + eval_fp) * (eval_tn + eval_fn))
    elif task == "qqp" or task == "mrpc":
        eval_epoch_metric2 = (2 * eval_tp) / (2 * eval_tp + eval_fn + eval_fp)
    else:
        eval_epoch_metric2 = eval_correct_cnt / eval_total_cnt

    # compute mean and std of target probability
    target_probability = torch.cat(target_probability)
    target_prob_std = target_probability.std().item()
    target_prob_mean = target_probability.mean().item()

    return {
        f"eval.loss": eval_epoch_loss.item(),
        f"eval.ppl": eval_ppl.item(),
        f"eval.acc": eval_epoch_acc * 100,
        f"eval.metric2": eval_epoch_metric2 * 100,
        f"eval.prob_std": target_prob_std,
        f"eval.prob_mean": target_prob_mean,
    }, target_probability.cpu().numpy()


def test_model(model, device, test_dataloader, tokenizer, PE_num=20, task="sst2"):
    model.eval()
    predictions = []
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            greedy_pred = torch.argmax(outputs.logits, dim=-1)
            decoded_output = tokenizer.batch_decode(
                greedy_pred[:, PE_num:], skip_special_tokens=True
            )
            for do in decoded_output:
                if task == "qqp" or task == "mrpc":
                    try:
                        if do.split()[0] == "no":
                            predictions.append(0)
                        else:
                            predictions.append(1)
                    except:
                        predictions.append(0)
                elif task == "rte" or task == "qnli":
                    try:
                        if do.split()[0] == "yes":
                            predictions.append(0)
                        else:
                            predictions.append(1)
                    except:
                        predictions.append(0)
                elif task == "mnli" or task == "mnli_matched" or task == "mnli_mismatched":
                    try:
                        if do.split()[0] == "yes":
                            predictions.append(0)
                        elif do.split()[0] == "maybe":
                            predictions.append(1)
                        else:
                            predictions.append(2)
                    except:
                        predictions.append(0)
                elif task == "sst2":
                    if do.split()[0] == "positive":
                        predictions.append(1)
                    else:
                        predictions.append(0)
                elif task == "cola":
                    if do.split()[0] == "acceptable":
                        predictions.append(1)
                    else:
                        predictions.append(0)
    return predictions


def evaluator_vis(logits, labels):
    target_prob = get_target_prob_from_logits_vis(logits, labels)
    pred = torch.argmax(logits, dim=-1)
    correct_cnt = sum(pred == labels).item()
    total_cnt = logits.shape[0]
    return target_prob, correct_cnt, total_cnt


def eval_model_vis(model, device, eval_dataloader):
    model.eval()
    eval_loss = 0
    eval_total_cnt = 0
    eval_correct_cnt = 0
    target_probability = []
    for step, batch in tqdm(enumerate(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        eval_loss += loss.detach().float()

        target_prob, correct_cnt, total_cnt = evaluator_vis(outputs.logits, batch["labels"])

        # collect target_probability for later analysis
        target_probability.append(target_prob)
        eval_total_cnt += total_cnt
        eval_correct_cnt += correct_cnt
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_epoch_acc = eval_correct_cnt / eval_total_cnt

    target_probability = torch.cat(target_probability)
    target_prob_std = target_probability.std().item()
    target_prob_mean = target_probability.mean().item()

    return {
        f"eval.loss": eval_epoch_loss.item(),
        f"eval.acc": eval_epoch_acc * 100,
        f"eval.prob_std": target_prob_std,
        f"eval.prob_mean": target_prob_mean,
    }, target_probability.cpu().numpy()


def eval_model_vis_fs(model, criterion, device, eval_dataloader):
    model.eval()
    eval_loss = 0
    eval_total_cnt = 0
    eval_correct_cnt = 0
    target_probability = []
    for step, batch in tqdm(enumerate(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            logits = model(batch["pixel_values"])
            loss = criterion(logits, batch["labels"])

        eval_loss += loss.detach().float()

        target_prob, correct_cnt, total_cnt = evaluator_vis(logits, batch["labels"])

        # collect target_probability for later analysis
        target_probability.append(target_prob)
        eval_total_cnt += total_cnt
        eval_correct_cnt += correct_cnt
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_epoch_acc = eval_correct_cnt / eval_total_cnt

    target_probability = torch.cat(target_probability)
    target_prob_std = target_probability.std().item()
    target_prob_mean = target_probability.mean().item()

    return {
        f"eval.loss": eval_epoch_loss.item(),
        f"eval.acc": eval_epoch_acc * 100,
        f"eval.prob_std": target_prob_std,
        f"eval.prob_mean": target_prob_mean,
    }, target_probability.cpu().numpy()


def eval_model_vis_fs_ae(model, criterion, device, eval_dataloader):
    model.eval()
    eval_loss = 0
    for step, batch in tqdm(enumerate(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            logits = model(batch["pixel_values"])
            loss = criterion(logits, batch["pixel_values"].view(logits.size(0), -1)) / logits.size(0)

        eval_loss += loss.detach().float()

    eval_epoch_loss = eval_loss / len(eval_dataloader)

    return {
        f"eval.loss": eval_epoch_loss.item()
    }


def get_target_prob_from_logits_vis(logits, labels):
    prob_logits = torch.nn.functional.softmax(logits, dim=-1).detach()
    label_prob = prob_logits[torch.arange(logits.shape[0]), labels]
    return label_prob


def data_collator(features):
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                # batch[k] = torch.tensor([f[k] for f in features])
                batch[k] = pad_sequence([torch.tensor(f[k]) for f in features], batch_first=True, padding_value=-100)

    return batch


def get_target_prob_from_logits(logits, labels):
    # recover target probability for every sample
    prob_logits = torch.nn.functional.softmax(logits, dim=-1).detach()
    batch_idx = []
    time_idx = []
    cls_idx = []
    if len(prob_logits.shape) == 3:
        for i in range(labels.shape[0]):
            batch_idx.append(i)
            time_idx.append(prob_logits.shape[1] - 2) # assume the second-to-last token corresponds to probability
            cls_idx.append(labels[i, 0])
        label_prob = prob_logits[batch_idx, time_idx, cls_idx].detach()
    elif len(prob_logits.shape) == 2:
        for i in range(labels.shape[0]):
            batch_idx.append(i)
            cls_idx.append(labels[i])
        label_prob = prob_logits[batch_idx, cls_idx].detach()
    return label_prob.cpu()

def remove_J_C_duplicates(iJ, thresh=0.95, retain_C_thresh=0.9):
    # sort all grads
    n = iJ.shape[0]
    iJ_norm_sqr = iJ.norm(dim=1) ** 2
    sorted_idx = torch.argsort(iJ_norm_sqr, descending=True)
    iJ = iJ[sorted_idx, :]
    iJ_norm_sqr = iJ_norm_sqr[sorted_idx]
    cumsum_iJ_norm_sqr = iJ_norm_sqr.cumsum(dim=0)
    for cut_idx, cs in enumerate(cumsum_iJ_norm_sqr):
        if cs / cumsum_iJ_norm_sqr[-1] > retain_C_thresh:
            break
    if cut_idx < n-1:
        iJ = iJ[:cut_idx+1, :]
        print("Pruned {:.1f}% of gradients".format((1 - cut_idx/n)*100))

    # compute correlation matrix
    iC = iJ @ iJ.T
    iJ_norm_sqr = iC.diag()
    cov_norm = (
        iJ_norm_sqr.unsqueeze(dim=-1) ** 0.5 @ iJ_norm_sqr.unsqueeze(dim=0) ** 0.5
    )
    cC = iC / cov_norm
    # find duplicate directions
    new_iJ_norm_sqr = iJ_norm_sqr
    remove_idx = [] # record vectors' idx to be removed
    actual_duplicate_idx = [] # for debug purpose, record actually duplicate idx
    SIM_THRESH = thresh
    iJ_rows = iC.shape[0]
    for i in range(iJ_rows):
        if i in actual_duplicate_idx:
            continue # do not investigate already removed idx
        for j in range(i+1, iJ_rows):
            if j in actual_duplicate_idx:
                continue # do not investigate already removed idx
            if torch.abs(cC[i, j]) > SIM_THRESH:
                # for duplicate directions, modify J and C accordingly
                remove_idx.append(j)
                if iJ_norm_sqr[i] == iJ_norm_sqr[j]:
                    actual_duplicate_idx.append(j)
                # accumulate the norm of j to i vector
                new_iJ_norm_sqr[i] += iJ_norm_sqr[j]
    if len(remove_idx) == 0:
        print("No Duplicate Directions")
        return iJ, iC
    else:
        # re-normalise with new_iJ_norm_sqr
        for i in range(iJ_rows):
            if new_iJ_norm_sqr[i] > iJ_norm_sqr[i]:
                scale_up = (new_iJ_norm_sqr[i] / iJ_norm_sqr[i])**0.5
                iC[i, :] *= scale_up
                iC[:, i] *= scale_up
                iJ[i, :] *= scale_up  
        # clean up duplicate directions
        grad_idx = [i for i in range(iC.shape[0]) if i not in remove_idx]
        iC = iC[grad_idx, :][:, grad_idx].clone()
        iJ = iJ[grad_idx, :]
        print("{:.3f}% Duplicate Directions, ({:.3f}% Actually Duplicate)".format(len(remove_idx)/iJ_rows*100, len(actual_duplicate_idx)/iJ_rows*100))
        return iJ, iC     