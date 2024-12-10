from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from peft import (
    get_peft_config,
    get_peft_model,
    PromptTuningInit,
    PromptTuningConfig,
    TaskType,
    PeftType,
    PeftModel,
    PeftConfig,
)
import torch
from datasets import load_dataset
import os
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from datasets import load_dataset
import random
import numpy as np
import wandb
import math
from iEFOptimizer import *
from glue_sst2_utils import *


def vHvp(scalar_output, target_params, v):
    """
    Assume a forward is run on the model which generated a scalar_output,
    this function will compute the Hessian * vector output, where the hessian is the curvature matrix of the scalar_output w.r.t. model parameter

    Inputs:
        v: a sequence of Tensors: Tensor sequence that match the dimension of target_params
        scalar_output: torch scalar: it connects to the computation graph of the model
        target_params: a sequence of Tensors: Tensor sequence that is connected to scalar_output, where Hvp is meant for

    Outputs:
    in param.grad
    """
    # acquire gradient for scalar_output w.r.t. parameters, with graph retained
    jac = torch.autograd.grad(scalar_output, target_params, create_graph=True)

    # interestingly, some tensors that has no second-order term would be set to no grad_fn in g
    # a safe guard is implemented here to avoid that issue: https://github.com/pytorch/pytorch/issues/73137
    for i, (grad, param) in enumerate(zip(jac, target_params)):
        if grad.grad_fn is None:
            grad += 0 * param

    # acquire Hv
    Hv = torch.autograd.grad(jac, target_params, v, retain_graph=True)
    vHv = vDp(v, Hv)
    return vHv


def vHvp_multi(scalar_output, target_params, v):
    """
    Assume a forward is run on the model which generated a scalar_output,
    this function will compute the Hessian * vector output, where the hessian is the curvature matrix of the scalar_output w.r.t. model parameter

    Inputs:
        v: a seqeuence of Tensors: each Tensor match the dimension of target_params
        scalar_output: torch scalar: it connects to the computation graph of the model
        target_params: a single Tensor: single Tensor  that is connected to scalar_output, where Hvp is meant for

    Outputs:
        a list of scalers
    """
    # acquire gradient for scalar_output w.r.t. parameters, with graph retained
    jac = torch.autograd.grad(scalar_output, target_params, create_graph=True)

    # interestingly, some tensors that has no second-order term would be set to no grad_fn in g
    # a safe guard is implemented here to avoid that issue: https://github.com/pytorch/pytorch/issues/73137
    for i, (grad, param) in enumerate(zip(jac, target_params)):
        if grad.grad_fn is None:
            grad += 0 * param

    # acquire Hv
    vHvs = []
    for vi in v:
        Hv = torch.autograd.grad(jac, target_params, v, retain_graph=True)
        vHvs.append(vDp(vi, Hv))
    return vHvs


def vGvp_multi(loss, logits, target_params, v):
    # acquire jacobian for loss w.r.t. logits, with graph retained
    loss_jac = torch.autograd.grad(loss, logits, create_graph=True)

    # acquire jacobian for logits w.r.t. target_params
    t = torch.ones_like(logits, requires_grad=True)

    # compute Jv = d Jtv / d t
    Jt = torch.autograd.grad(logits, target_params, t, create_graph=True)

    # acquire J H_l J v
    vGvs = []
    for vi in v:
        # calculate Jv
        Jv = torch.autograd.grad(Jt, t, vi, retain_graph=True)[0]

        # calculate H_l Jv
        HJv = torch.autograd.grad(loss_jac, logits, Jv, retain_graph=True)[0]

        # calculate J H_l Jv
        vGvs.append(vDp(Jv, HJv))
    return vGvs


def vDp(v1, v2):
    vDp_output = 0
    if type(v1) is not list:
        v1 = list(v1)
    if type(v2) is not list:
        v2 = list(v2)
    for v1i, v2i in zip(v1, v2):
        vDp_output += (v1i * v2i).sum().item()
    return vDp_output


def vHv_vg_multi(vHvs, vs, g):
    outputs = []
    for vHv, v in zip(vHvs, vs):
        vg = vDp(v, g)
        if vHv < 0:
            out = -1
        else:
            out = vHv**0.5 / vg
        outputs.append(out)
    return np.array(outputs)

def vHv_vg_multi_icml(vHvs, vs, g):
    outputs = []
    vgs = []
    for vHv, v in zip(vHvs, vs):
        vg = vDp(v, g)
        if vHv < 0:
            out = -1
        else:
            out = vHv**0.5 / vg
        outputs.append(out)
        vgs.append(vg)
    return np.array(outputs), np.array(vgs)


def multi_grad_compute(Js, logits_grads, damping=1e-7):
    """
    Compute the iEF, iEF(lg), EF, SGD and Adam gradient from a list of Jacobian matrix Js. In Js, every J's first dimension is assumed to be the batch dimension.
    """
    # stack all Js
    J_org = torch.cat(Js, dim=0)
    J = J_org.view(J_org.shape[0], -1)

    # compute logit_grad norm
    J_lg_org = torch.cat(logits_grads, dim=0)
    Jl = J_lg_org.view(J_lg_org.shape[0], -1)
    grad_logits_norms = Jl.norm(dim=1).to(torch.float64)

    # compute the covariance of J
    grad_inputs_norms = J.norm(dim=1).to(torch.float64)
    C = J @ J.T
    C = C.to(torch.float64)

    # add damping to covariance
    damping_matrix = torch.diag(C.diag())
    damped_C = (1 - damping) * C + damping * damping_matrix
    # Ci = torch.inverse((1 - damping) * C + damping * damping_matrix)

    # compute SGD grad
    SGD_grad = J.sum(dim=0).view(J_org.shape[1:])

    # compute Adam grad
    per_param_norm = (J**2).sum(dim=0) ** 0.5
    per_param_iEF_norm = ((J / grad_inputs_norms.unsqueeze(dim=-1)) ** 2).sum(
        dim=0
    ) ** 0.5
    iAdam_grad = SGD_grad / (per_param_iEF_norm.view(SGD_grad.shape) + damping)
    Adam_grad = SGD_grad / (per_param_norm.view(SGD_grad.shape) + damping)
    iAdam2_grad = SGD_grad / (per_param_iEF_norm.view(SGD_grad.shape) ** 2 + damping)
    Adam2_grad = SGD_grad / (per_param_norm.view(SGD_grad.shape) ** 2 + damping)

    # compute iEF grad
    target_grad_weighting = torch.linalg.solve(damped_C, (grad_inputs_norms**2)).to(torch.float32)
    # target_grad_weighting = ((grad_inputs_norms**2) @ Ci).to(torch.float32)
    grad_flat = target_grad_weighting @ J
    iEF_grad = grad_flat.view(J_org.shape[1:])

    # compute iEF(lg) grad
    target_grad_weighting = torch.linalg.solve(damped_C, (grad_logits_norms**2)).to(torch.float32)
    # target_grad_weighting = ((grad_logits_norms**2) @ Ci).to(torch.float32)
    grad_flat = target_grad_weighting @ J
    iEFlg_grad = grad_flat.view(J_org.shape[1:])

    # compute EF grad
    target_grad_weighting = torch.linalg.solve(damped_C, (grad_inputs_norms**0)).to(torch.float32)
    # target_grad_weighting = ((grad_inputs_norms**0) @ Ci).to(torch.float32)
    grad_flat = target_grad_weighting @ J
    EF_grad = grad_flat.view(J_org.shape[1:])

    # output_grads
    return (
        SGD_grad,
        EF_grad,
        iEF_grad,
        iEFlg_grad,
        iAdam_grad,
        Adam_grad,
        iAdam2_grad,
        Adam2_grad,
    ), ("SGD", "EF", "iEF", "iEFlg", "iAdam", "Adam", "iAdam2", "Adam2")

def save_grad(
    epoch_no, args, model, PE_module, train_dataloader, eval_dataloader, tokenizer
):
    # initialise saved grad expdir
    saved_grad_file_name = os.path.join(
        args.expdir, "grad_for_epoch_{}".format(epoch_no)
    )
    saved_grad_txt_file_name = os.path.join(args.expdir, "info_log.txt")
    if not os.path.exists(args.expdir):
        os.makedirs(args.expdir)

    # prepare container
    grad_dict = {}
    grad_dict["train_grad"] = []
    grad_dict["eval_grad"] = []

    # start recording text
    try:
        with open(saved_grad_txt_file_name, "a") as f:
            f.write("******************************************\n")
            f.write("*========================================*\n")
            f.write("*========================================*\n")
            f.write("******************************************\n")
            f.write("Start recording grad for epoch no {} \n".format(epoch_no))
    except:
        with open(saved_grad_txt_file_name, "w") as f:
            f.write("******************************************\n")
            f.write("*========================================*\n")
            f.write("*========================================*\n")
            f.write("******************************************\n")
            f.write("Start recording grad for epoch no {} \n".format(epoch_no))

    # setup training
    model.eval()  # always use eval grad

    # register module parameter
    target_params = list(PE_module.parameters())[0]

    # ready to collect gradients for train data
    train_epoch_loss = 0
    train_epoch_total_cnt = 0
    train_epoch_correct_cnt = 0
    accumulate_cnt = 0
    PE_module.saved_grad_inputs = []
    for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        batch = {k: v.to(args.device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        outputs.logits.retain_grad()
        loss.backward(retain_graph=True)
        accumulate_cnt += 1
        # compute vHv / vg
        logits_grad = outputs.logits.grad
        this_param_grad = PE_module.saved_grad_inputs[0]
        output_grads, grads_name = multi_grad_compute(
            [this_param_grad], [logits_grad], damping=1e-7
        )
        vHv_SGD = vHvp_multi(loss, target_params, [output_grads[0]])[0]
        vGvs = vGvp_multi(loss, outputs.logits, target_params, output_grads)
        vGv_SGD = vGvs[0]
        Rayleighs = vHv_vg_multi(vGvs, output_grads, output_grads[0])
        Rayleighs_rel = Rayleighs / Rayleighs[0]
        # write result to text file
        if Rayleighs[0] > 0:
            with open(saved_grad_txt_file_name, "a") as f:
                output_string = "Batch cnt {} has Rayleighs:".format(step)
                for i, gn in enumerate(grads_name):
                    if i == 0:
                        output_string += " SGD has G fitness: {:.5f}".format(
                            abs(vHv_SGD - vGv_SGD) / abs(vGv_SGD)
                        )
                    else:
                        output_string += " {}: {:.5f} |".format(gn, Rayleighs_rel[i])
                output_string += "\n"
                f.write(output_string)
        else:
            with open(saved_grad_txt_file_name, "a") as f:
                f.write("Batch cnt {} has invalid SGD rayleigh\n".format(step))

        # collect grad to cpu
        grad_dict["train_grad"].append(PE_module.saved_grad_inputs[0].cpu().numpy())
        PE_module.saved_grad_inputs = []
        # collect stats
        train_epoch_loss += loss.detach().float()
        target_prob, correct_cnt, total_cnt = evaluator(
            outputs.logits,
            batch["labels"],
            tokenizer,
        )
        train_epoch_total_cnt += total_cnt
        train_epoch_correct_cnt += correct_cnt
    # collect final stats for train
    train_epoch_loss /= accumulate_cnt
    train_epoch_acc = train_epoch_correct_cnt / train_epoch_total_cnt * 100
    # record stats
    grad_dict["train_grad"] = np.concatenate(grad_dict["train_grad"])
    grad_dict["train_loss"] = train_epoch_loss.item()
    grad_dict["train_epoch_acc"] = train_epoch_acc
    with open(saved_grad_txt_file_name, "a") as f:
        f.write(
            "Train Loss: {:5f}, Train Acc: {:.5f}%\n".format(
                train_epoch_loss, train_epoch_acc
            )
        )

    # ready to collect gradients for eval data
    eval_epoch_loss = 0
    eval_epoch_total_cnt = 0
    eval_epoch_correct_cnt = 0
    accumulate_cnt = 0
    for step, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
        batch = {k: v.to(args.device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        accumulate_cnt += 1
        # collect grad to cpu
        grad_dict["eval_grad"].append(PE_module.saved_grad_inputs[0].cpu().numpy())
        PE_module.saved_grad_inputs = []
        # collect stats
        eval_epoch_loss += loss.detach().float()
        target_prob, correct_cnt, total_cnt = evaluator(
            outputs.logits,
            batch["labels"],
            tokenizer,
        )
        eval_epoch_total_cnt += total_cnt
        eval_epoch_correct_cnt += correct_cnt
    # collect final stats for train
    eval_epoch_loss /= accumulate_cnt
    eval_epoch_acc = eval_epoch_correct_cnt / eval_epoch_total_cnt * 100
    # record stats
    grad_dict["eval_grad"] = np.concatenate(grad_dict["eval_grad"])
    grad_dict["eval_loss"] = eval_epoch_loss.item()
    grad_dict["eval_epoch_acc"] = eval_epoch_acc
    with open(saved_grad_txt_file_name, "a") as f:
        f.write(
            "Eval Loss: {:5f}, Eval Acc: {:.5f}%\n".format(
                eval_epoch_loss, eval_epoch_acc
            )
        )

    # save grads
    np.save(saved_grad_file_name, grad_dict)
    PE_module.saved_grad_inputs = []
