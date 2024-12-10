import math, time

import torch
import torch.optim as optim
from scipy.optimize import minimize, LinearConstraint
import numpy as np

def cosine_similarity(a, b):
    if a is None or b is None:
        return None, None, None
    abdp = 0
    anorm = 0
    bnorm = 0
    if type(a) is list and type(b) is list:
        for i, j in zip(a, b):
            abdp += (i * j).sum().item()
            anorm += (i**2).sum().item()
            bnorm += (j**2).sum().item()
    else:
        abdp += (a * b).sum().item()
        anorm += (a**2).sum().item()
        bnorm += (b**2).sum().item()
    if anorm == 0 or bnorm == 0:
        return 0, 0, 0
    anorm = anorm**0.5
    bnorm = bnorm**0.5
    return abdp / anorm / bnorm, anorm, bnorm


SVD_TRUNCATE_N = 500


class generalPurposeiEFOptimiser(optim.Optimizer):
    """
    This is an optimiser designed specifically for prompt tuning
    """
    def flat_grad_to_grad_list(self, flat_grad):
        """convert a 1-dimensional grad to a usable parameter gradient"""
        assert flat_grad.numel() == self.trainable_parameter_size
        grad_list = []
        flat_grad_idx = 0
        for param_id, param in enumerate(self.trainable_parameters_sequence):
            this_param_size = param.numel()
            this_grad = flat_grad[flat_grad_idx:flat_grad_idx+this_param_size]
            this_grad = torch.reshape(this_grad, param.shape)
            grad_list.append(this_grad.clone())
            flat_grad_idx += this_param_size
        return grad_list

    def grad_list_to_flat_grad(self, grad_list):
        """convert a default parameter grad list to a 1-dimensional grad"""
        flat_grad_list = [grad.view(1, -1) for grad in grad_list]
        flat_grad = torch.cat(flat_grad_list, dim=-1)[0, :]
        assert flat_grad.numel() == self.trainable_parameter_size
        return flat_grad
    
    def clone_grad_list(self, grad_list):
        """return a grad_list, with every element the clone of the provided grad_list"""
        return [grad.clone() for grad in grad_list]
    
    def zero_grad_list(self, grad_list_like):
        """return a grad_list where every tensor is all-zeros. shaped as grad_list_like"""
        return [torch.zeros_like(grad) for grad in grad_list_like]
    
    def norm_grad_list(self, grad_list):
        """return a float type, indicating the norm of the grad_list"""
        sqr_norm = 0
        for grad in grad_list:
            sqr_norm += (grad**2).sum().item()
        return sqr_norm ** 0.5
    
    def dot_grad_list(self, grad_list1, grad_list2):
        """return the dot product of two given grad_list's"""
        dot_out = 0
        for g1, g2 in zip(grad_list1, grad_list2):
            dot_out += (g1 * g2).sum().item()
        return dot_out
    
    def add_grad_list(self, grad_list1, grad_list2=None, alpha=1, beta=1, out=None):
        """return out, where out = grad_list1 * alpha + grad_list2 * beta
        when grad_list2 is None or beta=0, then out = grad_list1 * alpha
        """
        if out is None:
            out = []
            if grad_list2 is not None:
                assert len(grad_list1) == len(grad_list2)
                for g1, g2 in zip(grad_list1, grad_list2):
                    out.append(g1 * alpha + g2 * beta)
            else:
                for g1 in grad_list1:
                    out.append(g1 * alpha)
        else:
            assert len(out) == len(grad_list1)
            if grad_list2 is not None:
                assert len(grad_list1) == len(grad_list2)
                for i, (g1, g2) in enumerate(zip(grad_list1, grad_list2)):
                    out[i] = g1 * alpha + g2 * beta
            else:    
                for i, g1 in enumerate(grad_list1):
                    out[i] = g1 * alpha
        return out
    
    def clear_saved_grad(self, SF=False):
        for module in self.recognised_trainable_modules:
            if not SF:
                module.sf_activation = None
                module.activation = []
                module.sf_error_signal = []
                module.error_signal = []
                module.J = []
                module.sf_J = []
            else:
                module.sf_activation = None
                module.error_signal = []
                module.J = []
        torch.cuda.empty_cache()

    # disable per-sample grad collection
    def disable_persample_grad(self):
        for module in self.recognised_trainable_modules:
            module.enable_persample = False

    # enable per-sample grad collection
    def enable_persample_grad(self):
        for module in self.recognised_trainable_modules:
            module.enable_persample = True

    # for fisher sampling
    def _switch_to_SF_mode(self):
        # set modules to record SF jacobian
        for module in self.recognised_trainable_modules:
            module.error_signal, module.sf_error_signal = module.sf_error_signal, module.error_signal
            module.J, module.sf_J = module.sf_J, module.J
            module.inSF = True
        for param in self.trainable_parameters_sequence:
            param.backup_grad = param.grad
            param.grad = None

    # for fisher sampling
    def _switch_from_SF_mode(self):
        # set modules to record standard SF jacobian
        for module in self.recognised_trainable_modules:
            module.error_signal, module.sf_error_signal = module.sf_error_signal, module.error_signal
            module.J, module.sf_J = module.sf_J, module.J
            module.inSF = False
        for param in self.trainable_parameters_sequence:
            param.grad, param.backup_grad = param.backup_grad, param.grad

    def collect_SF_Jacobian(self):
        """
        Collate self.SF_Js Jacobians into one large Jacobian
        """
        large_SF_J = torch.cat(self.SF_Js, dim=0)
        self.SF_Js = []
        return large_SF_J
    
    def compute_Jacobian(self, SF=False):
        """
        Collect gradients for all modules and compute the Jacobian
        When in standard mode (SF = False):
            For each module, compute per-sample gradient based on module.activations and module.error_signal (or module.J)
            Collect per-sample gradient into a large Jacobian
            Clear everything in module (including sf related information, so make sure you collect SF grad before calling this function)
        When in SF (fisher sampling mode)
            For each module, assume the module.error_signal/J is already swapped with sf_error_signal/sf_J
            Compute per-sample gradient based on single module.sf_activation and multiple module.error_signal/J
            Collect per-sample gradient into a large Jacobian
            Clear only sf related information, which is currectly stored in module.error_signal/J (standard batch grad information are stored in sf_error_signal/sf_J due to the swap)
        """
        assert self.recognised_trainable_modules[0].enable_persample
        # compute number of samples computed
        batch_cnt = len(self.recognised_trainable_modules[0].error_signal)
        sample_cnt = 0
        for i, e in enumerate(self.recognised_trainable_modules[0].error_signal):
            if e is not None:
                sample_cnt += e.shape[0]
                device = e.device
            else:
                eJ = self.recognised_trainable_modules[0].J[i]
                sample_cnt += eJ[0].shape[0]
                device = eJ[0].device

        # create Jacobian placeholder
        J = torch.zeros((sample_cnt, self.trainable_parameter_size), device=device, dtype=torch.float32)

        # compute per-sample grad for every parameters
        total_param_idx = 0
        for module in self.recognised_trainable_modules:
            sample_idx = 0
            classname = module.__class__.__name__
            if classname == "Linear":
                for b in range(batch_cnt):
                    if module.J[b] is None:
                        e = module.error_signal[b]
                        n = e.shape[0]
                        this_param_idx = total_param_idx
                        if module.weight.requires_grad:
                            if not SF:
                                a = module.activation[b]
                            else:
                                a = module.sf_activation
                            if len(a.shape) == 3 and len(a.shape) == 3:
                                eW = torch.einsum("abi,abj->aij",e, a)
                            elif len(a.shape) == 2 and len(a.shape) == 2:
                                eW = torch.einsum("ai,aj->aij", e, a)
                            else:
                                raise ValueError("Unknown input output dimension for this Linear module")
                            gW = eW.view(n, -1)
                            J[sample_idx:sample_idx+n, this_param_idx:this_param_idx+gW.shape[1]] = gW
                            this_param_idx += gW.shape[1]
                        if module.bias is not None and module.bias.requires_grad:
                            if len(e.shape) == 3:
                                eb = e.sum(dim=1)
                            elif len(e.shape) == 2:
                                eb = e.clone()
                            else:
                                raise ValueError("Unknown input output dimension for this Linear module")
                            gb = eb.view(n, -1)
                            J[sample_idx:sample_idx+n, this_param_idx:this_param_idx+gb.shape[1]] = gb
                            this_param_idx += gb.shape[1]
                        sample_idx += n
                    else: # per-sample gradient already computed
                        this_J = module.J[b]
                        n = this_J[0].shape[0]
                        this_param_idx = total_param_idx
                        Jidx = 0
                        if module.weight.requires_grad:
                            gW = this_J[Jidx]
                            J[sample_idx:sample_idx+n, this_param_idx:this_param_idx+gW.shape[1]] = gW
                            this_param_idx += gW.shape[1]
                            Jidx += 1
                        if module.bias is not None and module.bias.requires_grad:
                            gb = this_J[Jidx]
                            J[sample_idx:sample_idx+n, this_param_idx:this_param_idx+gb.shape[1]] = gb
                            this_param_idx += gb.shape[1]
                        module.J[b] = None
                        sample_idx += n
                total_param_idx = this_param_idx
                gW = None; eW = None; gb = None; eb = None; e = None; a = None
            elif classname == "Conv2d":
                for b in range(batch_cnt):
                    e = module.error_signal[b]
                    n = e.shape[0]
                    this_param_idx = total_param_idx
                    if module.weight.requires_grad:
                        if not SF:
                            a = module.activation[b]
                        else:
                            a = module.sf_activation
                        a = torch.nn.functional.unfold(a, kernel_size=module.kernel_size, dilation=module.dilation, padding=module.padding, stride=module.stride)
                        aT = a.transpose(1, 2)
                        e = e.reshape(n, -1, a.shape[-1])
                        grad1 = torch.bmm(e, aT)
                        # s = time.time()
                        # for i in range(10000):
                        #     grad_ein = torch.einsum('ijk,ilk->ijl', e, a)
                        # eint = time.time() - s
                        # s = time.time()
                        # for i in range(10000):
                        #     aT = a.transpose(1, 2)
                        #     grad_bmm = torch.bmm(e, aT)
                        # bmmt = time.time() - s
                        shape = [n] + list(module.weight.shape)
                        eW = grad1.reshape(shape)
                        gW = eW.view(n, -1)
                        J[sample_idx:sample_idx+n, this_param_idx:this_param_idx+gW.shape[1]] = gW
                        this_param_idx += gW.shape[1]
                    if module.bias is not None and module.bias.requires_grad:
                        # !!! THERE IS HUGE ERROR COMPARED TO STANDARD BACKPROP, PLEASE CHECK NEXT TIME
                        raise NotImplementedError
                        eb = torch.sum(e, dim=2)
                        gb = eb.view(n, -1)
                        J[sample_idx:sample_idx+n, this_param_idx:this_param_idx+gb.shape[1]] = gb
                        this_param_idx += gb.shape[1]
                    sample_idx += n
                total_param_idx = this_param_idx
                gW = None; eW = None; gb = None; eb = None; e = None; a = None
            elif classname == "BatchNorm2d":
                for b in range(batch_cnt):
                    e = module.error_signal[b]
                    if not SF:
                        x = module.activation[b]
                    else:
                        x = module.sf_activation
                    n = x.shape[0]
                    if module.training:
                        xmean = x.mean((0, 2, 3), keepdim=True) # batch mean
                        xvar = x.var((0, 2, 3), keepdim=True, unbiased=False) # batch variance
                    else:
                        xmean = module.running_mean
                        xvar = module.running_var
                    xhat = (x - xmean) / (xvar + module.eps)**0.5
                    eW = (e * xhat).sum(dim=(-1, -2))
                    eb = e.sum(dim=(-1, -2))
                    gW = eW.view(n, -1)
                    gb = eb.view(n, -1)
                    J[sample_idx:sample_idx+n, total_param_idx:total_param_idx+gW.shape[1]] = gW
                    J[sample_idx:sample_idx+n, total_param_idx+gW.shape[1]:total_param_idx+gW.shape[1]+gb.shape[1]] = gb
                    sample_idx +=n 
                total_param_idx += gW.shape[1] + gb.shape[1]
                gW = None; eW = None; gb = None; eb = None; e = None; a = None
            elif classname == "PromptEmbedding":
                for b in range(batch_cnt):
                    e = module.error_signal[b]
                    n = e.shape[0]
                    e = e.view(n, -1)
                    J[sample_idx:sample_idx+n, total_param_idx:total_param_idx+e.shape[1]] = e
                    sample_idx +=n
                total_param_idx += e.shape[1]
                e = None

        # clear per module grad cache
        self.clear_saved_grad(SF=SF)

        # return Jacobian
        return J

    def __init__(
        self,
        model,
        lr=1e-3,
        clip_norm=None,
        momentum=0.9,
        momentum2=0.999,
        damping=1e-3,
        opt_flag="SGD",
        grad_renorm=False,
        norm_update=False,
        weight_decay=0,
        const_weight=False,
        norm_weight=False,
        max_samples=None,
        mean_truncate=None,
        adapt_lr=False,
        sort_with="grad_norm",
        converged_prob_thresh=None,
        cossim_bound=None,
        af_relative_step=True,
        af_warmup_init=False,
        af_scale_parameter=True,
        af_eps=[1e-30, 0.001],
        line_search=False,
        line_search_beta=0.1,
        line_search_alpha=0.001,
        memory_efficient_persample_gradient=True,
    ):
        defaults = dict(
            lr=lr,
            clip_norm=clip_norm,
            momentum=momentum,
            momentum2=momentum2,
            damping=damping,
            opt_flag=opt_flag,
            grad_renorm=grad_renorm,
            norm_update=norm_update,
            weight_decay=weight_decay,
            const_weight=const_weight,
            norm_weight=norm_weight,
            max_samples=max_samples,
            mean_truncate=mean_truncate,
            adapt_lr=adapt_lr,
            sort_with=sort_with,
            converged_prob_thresh=converged_prob_thresh,
            cossim_bound=cossim_bound,
            af_relative_step=af_relative_step,
            af_warmup_init=af_warmup_init,
            af_scale_parameter=af_scale_parameter,
            af_eps=af_eps,
            line_search=line_search,
            line_search_beta=line_search_beta,
            line_search_alpha=line_search_alpha,
        )
        """
        opt_flag: specify which version of the optimiser to use:
                    -1: SGD
                    0 ~ 3: EF related optimisers, 0 is TONGA, 2 is iEF, 1 is in-between EF, 3 is new iEF
                    4 ~ 5: Adam related optimisers, 4 is normal adam, 5 is adam but use per-sample gradient to estimate v
        """
        super(generalPurposeiEFOptimiser, self).__init__(
            model.parameters(), defaults
        )

        # get all trainable and recognised modules
        self.model = model
        self.known_modules = ["PromptEmbedding", "Linear", "Conv2d", "BatchNorm2d"] #"Linear"
        self.recognised_trainable_modules = []
        self.trainable_parameters_sequence = []
        self.trainable_parameter_size = 0

        # filter out all trainable modules, and make sure we recognise them all
        unknown_trainable_modules = []
        partially_trainable_recognised_modules = []
        for module_name, module in model.named_modules():
            classname = module.__class__.__name__
            # check if all parameters requires grad
            module_parameters = list(module.parameters())
            if len(module_parameters) == 0:
                continue
            all_require_grad = True
            any_require_grad = False
            for param in module_parameters:
                if param.requires_grad:
                    any_require_grad = True
                else:
                    all_require_grad = False
            if all_require_grad: 
                # first accept all fully trainable modules
                # this is to make sure it is easy to map parameters to modules
                if classname not in self.known_modules:
                    # keep note of unknown modules, and check with known modules to avoid duplication
                    unknown_trainable_modules.append((module_name, module))
                elif module not in self.recognised_trainable_modules: # avoid duplicate modules
                    self.recognised_trainable_modules.append(module)
                    for param in module_parameters:
                        self.trainable_parameters_sequence.append(param)
                        self.trainable_parameter_size += torch.numel(param)
            elif any_require_grad:
                if classname in self.known_modules:
                    partially_trainable_recognised_modules.append((module_name, module))
        for module_name, module in partially_trainable_recognised_modules:
            # this is to handle lora_only bias case, where a Linear module includes lora weights and original weight and bias. The original is the only untrainable parameter
            # now that we have added all fully trainable modules (i.e. handled lora's weight), we could add the final trainable "original bias"
            # this handling may not work for other partially trianable module cases!!!
            any_unique_parameter = False
            for param in module.parameters():
                if param.requires_grad and param not in set(self.trainable_parameters_sequence):
                    self.trainable_parameters_sequence.append(param)
                    self.trainable_parameter_size += torch.numel(param)
                    any_unique_parameter = True
            if any_unique_parameter:
                self.recognised_trainable_modules.append(module)
            
        # report module collection results
        print("There are {} trainable modules and {} trainable parameters".format(len(self.recognised_trainable_modules), self.trainable_parameter_size))
        # if trainable parameters are not all included in trainable modules, then there are unknown modules!
        true_trainable_parameter_size = sum([p.numel() for p in model.parameters() if p.requires_grad])
        if true_trainable_parameter_size != self.trainable_parameter_size:
            raise ValueError("Recogonised trainable modules only cover {:.3f}% all trainable parameters".format(self.trainable_parameter_size / true_trainable_parameter_size * 100))

        # add backward hook to known modules to record per-sample gradients
        for rtmodule in self.recognised_trainable_modules:
            classname = rtmodule.__class__.__name__
            # define different backward hook for different modules
            # these backward hook functions can retain per-sample gradient during backpropagation
            if classname == "PromptEmbedding":
                def save_forward_activation(module, input, output):
                    if not module.enable_persample:
                        return
                    pass
                def save_error_signal(module, grad_output, grad_input):
                    if not module.enable_persample:
                        return
                    module.error_signal.append(grad_input[0].detach())
            elif classname == "Conv2d":
                def save_forward_activation(module, input, output):
                    if not module.enable_persample:
                        return
                    if module.weight.requires_grad:
                        module.activation.append(input[0].detach())
                        module.sf_activation = input[0].detach()
                def save_error_signal(module, grad_output, grad_input):
                    if not module.enable_persample:
                        return
                    module.error_signal.append(grad_input[0].detach())
            elif classname == "BatchNorm2d":
                def save_forward_activation(module, input, output):
                    if not module.enable_persample:
                        return
                    module.activation.append(input[0].detach())
                    module.sf_activation = input[0].detach()
                def save_error_signal(module, grad_output, grad_input):
                    if not module.enable_persample:
                        return
                    module.error_signal.append(grad_input[0].detach())
            elif classname == "Linear":
                def save_forward_activation(module, input, output):
                    if not module.enable_persample:
                        return
                    if module.weight.requires_grad:
                        module.activation.append(input[0].detach())
                        module.sf_activation = input[0].detach()
                if not memory_efficient_persample_gradient:
                    def save_error_signal(module, grad_output, grad_input):
                        if not module.enable_persample:
                            return
                        module.error_signal.append(grad_input[0].detach())
                else:
                    def save_error_signal(module, grad_output, grad_input):
                        if not module.enable_persample:
                            return
                        module.error_signal.append(grad_input[0].detach())
                        # compute the per-sample gradient if activation or error_signal is too expensive to store
                        parameter_size = 0
                        storage_size = grad_input[0].numel()
                        n = grad_input[0].shape[0]
                        if module.weight.requires_grad:
                            parameter_size += module.weight.numel()
                            if not module.inSF:
                                storage_size += module.activation[-1].numel()
                        if module.bias is not None and module.bias.requires_grad:
                            parameter_size += module.bias.numel()
                        if storage_size < n * parameter_size:
                            module.J.append(None)
                        else:
                            # since storing the activation and error signal is too expensive
                            # we will compute the per-sample gradient right away
                            this_J = []
                            e = module.error_signal[-1]
                            if module.weight.requires_grad:
                                if not module.inSF:
                                    a = module.activation[-1]
                                else:
                                    a = module.sf_activation
                                if len(a.shape) == 3 and len(a.shape) == 3:
                                    eW = torch.einsum("abi,abj->aij",e, a)
                                elif len(a.shape) == 2 and len(a.shape) == 2:
                                    eW = torch.einsum("ai,aj->aij", e, a)
                                else:
                                    raise ValueError("Unknown input output dimension for this Linear module")
                                gW = eW.view(n, -1)
                                this_J.append(gW)
                                # free memory
                                if not module.inSF:
                                    module.activation[-1] = None
                            if module.bias is not None and module.bias.requires_grad:
                                if len(e.shape) == 3:
                                    eb = e.sum(dim=1)
                                elif len(e.shape) == 2:
                                    eb = e.clone()
                                else:
                                    raise ValueError("Unknown input output dimension for this Linear module")
                                gb = eb.view(n, -1)
                                this_J.append(gb)
                            # free memory
                            module.error_signal[-1] = None
                            module.J.append(this_J)

            # register forward and backward hook    
            rtmodule.register_forward_hook(save_forward_activation)
            rtmodule.register_full_backward_hook(save_error_signal)
            rtmodule.activation = []
            rtmodule.error_signal = []
            rtmodule.J = []
            rtmodule.sf_activation = []
            rtmodule.sf_error_signal = []
            rtmodule.sf_J = []
            rtmodule.inSF = False
            rtmodule.enable_persample = True
            for param in rtmodule.parameters():
                if param.requires_grad:
                    param.saved_persample_grad = []

        # stats
        self.update_time = 0
        self.update_count = 0
        self.prev_grad_inputs_matrix = None
        self.prev_old_grad = None
        self.prev_new_grad = None
        self.prev_update = None
        self.iEF_prev_grad = None
        self.momentum_grad = None
        self.adam_m = None
        self.adam_v = None
        self.af_row = None
        self.af_col = None
        self.logits_grad_norm = []
        self.max_loss_reduction = []
        self.max_logprob_improvement = []
        self.target_metric_mean = 0
        self.SF_Js = []

        # converged grad stats
        self.converged_sample_grad = []
        self.truncated_cl_grads = None
        self.truncated_unconv_grads = None
        self.truncated_iEF_matrix = None
        self.prev_unconverged_grads = None

        # norm analysis
        self.prob_history = []
        self.norm_history = []

    def __setstate__(self, state):
        super(generalPurposeiEFOptimiser, self).__setstate__(state)

    def update_logits_grad(self, logits_grad, loss, logits, labels):
        """
        Collect logits_grad
        """
        # record logits_grad_norm
        non_batch_dim = tuple(range(1, len(logits_grad.shape)))
        self.logits_grad_norm.append(logits_grad.norm(dim=non_batch_dim))

        # record label probability
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
        self.max_logprob_improvement.append(-torch.log(label_prob))

        # record grad * alpha * grad
        loss_jac = torch.autograd.grad(loss, logits, create_graph=True)
        Hg = torch.autograd.grad(loss_jac, logits, logits_grad, retain_graph=True)[0]
        gHg = (logits_grad * Hg).sum(dim=non_batch_dim)
        gg = logits_grad.norm(dim=non_batch_dim) ** 2
        alpha = gg / gHg
        self.max_loss_reduction.append(0.5 * alpha * gg)

    def fisher_sampling(self, logits, original_labels, MC_TIME=1, retain_graph=False):
        """
        This function samples the fisher from model output distribution for the current batch
        It is assumed that the forward operation is done for this batch
        Inputs:
            logits: Tensor: the output logits of the current batch, make sure a forward graph has already been computed for this tensor
            original_labels: Tensor: the true label for this batch
            MC_TIME: int: number of monte-carlo sampling required
            retrain_graph: boolean: whether to retain the computation graph, default False
        """
        tmp_time_st = time.time()
        # for this batch of data, sample the corresponding jacobian using model output distribution
        # convert PT logits to standard logits (remove starting tokens)
        if len(logits.shape) == 3 and logits.shape[1] > original_labels.shape[1]:
            logits = logits[:, -original_labels.shape[1]:, :].clone()
        # get output distribution
        logits_probs = torch.nn.functional.softmax(logits, dim=-1).detach().cpu()
        if len(logits_probs.shape) == 3:
            logits_probs = logits_probs[:, logits_probs.shape[1]-2, :]
        elif len(logits_probs.shape) == 2:
            logits_probs = logits_probs
        else:
            raise NotImplementedError
        
        # interestingly, the graph between the original logits and CE_loss is dependent on the label tensor, 
        # modifying the label tensor would disrupt further backward operations
        # that's why a clone of the original_label is made here
        original_labels = original_labels.to(logits.device).clone() 

        # compute center trace
        center_trace = ((1 + (logits_probs ** 2).sum(dim=1, keepdim=True)) - 2*logits_probs) * logits_probs
        max_trace, max_trace_label = torch.max(center_trace, dim=1, keepdim=True)

        # do custom operations on target probs
        batch_idx = []
        cls_idx = []
        for i in range(original_labels.shape[0]):
            batch_idx.append(i)
            if len(original_labels.shape) == 2:
                cls_idx.append(original_labels[i, 0])
        # logits_probs[batch_idx, cls_idx] = 0 # set target prob to 0, so samples are all generated from non-target labels

        # backup every parameters' grad and module's saved_grad_inputs
        self._switch_to_SF_mode()
        
        # do monte carlo sampling
        all_sampled_labels = []
        for mc_i in range(MC_TIME):
            # generate MC sampled label
            sampled_label = torch.multinomial(logits_probs, 1).detach()
            # sampled_label = torch.multinomial(center_trace, 1).detach()
            # sampled_label = max_trace_label
            all_sampled_labels.append(sampled_label.squeeze())

            # update batch label
            if len(original_labels.shape) == 1:
                original_labels[:] = sampled_label[:, 0]
            elif len(original_labels.shape) == 2:
                original_labels[:, 0:1] = sampled_label
            else:
                raise NotImplementedError
            
            # do backprop and get jacobian
            CE_loss = torch.nn.CrossEntropyLoss()
            flatten_logits = logits.view(-1, logits.shape[-1])
            flatten_labels = original_labels.view(-1)
            new_loss = CE_loss(flatten_logits, flatten_labels)
            new_CE_grad = torch.autograd.grad(new_loss, logits, retain_graph=True)[0]
            if mc_i == MC_TIME - 1:
                logits.backward(new_CE_grad, retain_graph=retain_graph) # last, graph can be destroyed on request
            else:
                logits.backward(new_CE_grad, retain_graph=True)

        # compute SF_J and store them
        self.SF_Js.append(self.compute_Jacobian(SF=True))

        # recover to standard saved grads
        self._switch_from_SF_mode()

        # record update time
        self.update_time += time.time() - tmp_time_st

        # analyse sampled labels' sample efficiency (i.e. after the MC_TIME sampling, how many unique gradients are given)
        all_sampled_labels = torch.stack(all_sampled_labels)
        unique_label_counts = 0
        for i in range(all_sampled_labels.shape[1]):
            ul = torch.unique(all_sampled_labels[:, i])
            unique_label_counts += len(ul)
        sample_efficiency = unique_label_counts / torch.numel(all_sampled_labels)

    def step(self, closure=None, batch_loss=None, debug_batch_data=None):
        # setup stats
        self.stats = {}
        # retrieve optimiser's hyperparameters: all parameters share the same hyperparameters
        group = self.param_groups[0]
        lr = group["lr"]
        clip_norm = group["clip_norm"]
        damping = group["damping"]
        momentum = group["momentum"]
        momentum2 = group["momentum2"]
        opt_flag = group["opt_flag"]
        grad_renorm = group["grad_renorm"]
        norm_update = group["norm_update"]
        weight_decay = group["weight_decay"]
        const_weight = group["const_weight"]
        norm_weight = group["norm_weight"]
        max_samples = group["max_samples"]
        mean_truncate = group["mean_truncate"]
        adapt_lr = group["adapt_lr"]
        sort_with = group["sort_with"]
        converged_prob_thresh = group["converged_prob_thresh"]
        cossim_bound = group["cossim_bound"]
        af_relative_step = group["af_relative_step"]
        af_warmup_init = group["af_warmup_init"]
        af_scale_parameter = group["af_scale_parameter"]
        af_eps = group["af_eps"]
        line_search = group["line_search"]
        line_search_beta = group["line_search_beta"]
        line_search_alpha = group["line_search_alpha"]
        # PE_param = group["params"][0]

        # step time cost
        self.update_time = 0

        # directly compute momentum grad (gradient estimation with momentum)
        param_grad_list = [param.grad for param in self.trainable_parameters_sequence]
        if "sf" not in opt_flag.lower() and opt_flag not in (
            "CLiEFx",
            "ConvGD",
            "ConvGDu",
            "ConviEF",
            "focalGD",
            "n(1-p)GD",
            "n(1-p)GDi",
            "n(1-p)2GD",
            "n(1-p)2GDi",
            "n(1-p2)GD",
            "n(1-p2)GDi",
            "convGD",
            "convGDi",
            "GD",
            "GDi",
        ):  # for special optimisers types, use custom momentum computation
            if self.momentum_grad is not None and momentum != 0:
                self.momentum_grad = self.add_grad_list(self.momentum_grad, param_grad_list, momentum, 1-momentum, out=self.momentum_grad)
            else:
                self.momentum_grad = self.clone_grad_list(param_grad_list)

        # increment update count
        self.update_count += 1
        if self.update_count == 1:
            self.init_weight_norm = self.norm_grad_list(self.trainable_parameters_sequence)

        # collect gradients for all modules
        J = self.compute_Jacobian()
        sample_cnt = J.shape[0]
        # compute norm stats for J
        grad_inputs_norms = J.norm(dim=1)

        # # debug 
        # sumJ = J.sum(dim=0)
        # mgrad = self.grad_list_to_flat_grad(param_grad_list)
        # gcs = cosine_similarity(sumJ, mgrad)
        # a = 0

        # collect information at logits level
        lgn = torch.cat(self.logits_grad_norm, dim=0)
        newton_lr = torch.cat(self.max_loss_reduction, dim=0)
        logp_lr = torch.cat(self.max_logprob_improvement, dim=0)
        target_log_prob = -logp_lr
        target_prob = torch.exp(target_log_prob)
        self.logits_grad_norm = []
        self.max_loss_reduction = []
        self.max_logprob_improvement = []

        # choose which per-sample stats to use to sort samples
        if sort_with == "grad_norm":
            target_metric = grad_inputs_norms
            descending = True
        elif sort_with == "target_prob":
            target_metric = target_prob
            descending = False
        elif sort_with == "random":
            target_metric = None
            mean_truncate = None  # random sort means no mean truncate
        else:
            raise ValueError("Unknown sort_with Argument: {}!!".format(sort_with))

        # sort gradients from smaller norm to larger norm
        if target_metric is not None:
            sorted_index = torch.argsort(target_metric, descending=descending)
        else:
            # assume samples per-batch are already shuffled
            sorted_index = torch.arange(sample_cnt)

        # resort per-sample stats according to sort_with indices
        J = J[sorted_index, :]
        grad_inputs_norms = grad_inputs_norms[sorted_index]
        lgn = lgn[sorted_index]
        newton_lr = newton_lr[sorted_index]
        logp_lr = logp_lr[sorted_index]
        target_prob = target_prob[sorted_index]

        # record prob and norm history
        self.prob_history.append(target_prob.detach().cpu().numpy())
        self.norm_history.append(J.norm(dim=1).detach().cpu().numpy())

        # target metric mean estimate
        if sort_with != "random":
            sample_cnt_5pt = int(sample_cnt * 0.05)
            target_metric = target_metric[sorted_index]
            this_target_metric_mean = target_metric[
                sample_cnt_5pt:-sample_cnt_5pt
            ].mean()
            self.target_metric_mean = (
                0.5 * self.target_metric_mean + 0.5 * this_target_metric_mean
            )

        # truncate sample according to mean norm
        if mean_truncate is not None:  # truncate gradients according to mean
            if sort_with == "random":
                raise ValueError("Cannot have mean_truncate when sorting with random.")
            retained_sample_cnt = (
                target_metric < self.target_metric_mean * mean_truncate
            ).sum()
            if retained_sample_cnt < max_samples:
                max_samples = retained_sample_cnt + 1

        # compress per-sample information
        if (
            "iEF" in opt_flag
            and max_samples is not None
            and max_samples + 1 < sample_cnt
        ):
            print("Sample Truncation in effect")
            accumulate_samples_cnt = sample_cnt - max_samples + 1
            sum_grad = J[:accumulate_samples_cnt, :].sum(dim=0, keepdim=True)
            remain_grad = J[accumulate_samples_cnt:, :]
            J = torch.cat((sum_grad, remain_grad), dim=0)
            sample_cnt_old = sample_cnt
            sample_cnt = max_samples

            # update scaling factors
            grad_inputs_norms_old = grad_inputs_norms
            grad_inputs_norms = J.norm(dim=1)

            # for logit grad norm, sum the collapsed ones together
            lgn_new = torch.zeros_like(grad_inputs_norms)
            lgn_new[0] = torch.sqrt((lgn[:accumulate_samples_cnt] ** 2).sum())
            lgn_new[1:] = lgn[accumulate_samples_cnt:]
            lgn_old = lgn
            lgn = lgn_new

            # for newton learning rate, take the sum of the collapsed ones
            newton_lr_new = torch.zeros_like(grad_inputs_norms)
            newton_lr_new[0] = newton_lr[:accumulate_samples_cnt].sum()
            newton_lr_new[1:] = newton_lr[accumulate_samples_cnt:]
            newton_lr_old = newton_lr
            newton_lr = newton_lr

            # for logp_lr, take the sum of the collapsed ones
            logp_lr_new = torch.zeros_like(grad_inputs_norms)
            logp_lr_new[0] = logp_lr[:accumulate_samples_cnt].sum()
            logp_lr_new[1:] = logp_lr[accumulate_samples_cnt:]
            logp_lr_old = logp_lr
            logp_lr = logp_lr_new

            # for p_lr, thake the mean of the collapsed ones
            p_lr_new = torch.zeros_like(grad_inputs_norms)
            p_lr_new[0] = target_prob[:accumulate_samples_cnt].mean()
            p_lr_new[1:] = target_prob[accumulate_samples_cnt:]
            p_lr_old = target_prob
            target_prob = p_lr_new

        # for iEF method, generate target_grad_norm
        target_floating_point = torch.float64
        target_loss_reduction = None
        prefix_iEF = ""
        if "iEF" in opt_flag and "CLiEF" not in opt_flag and "Conv" not in opt_flag:
            if opt_flag.startswith("flex"):
                prefix_iEF = "flex"
                opt_flag = opt_flag[4:]
            if opt_flag == "iEF-custom":
                # base scaling
                target_loss_reduction = grad_inputs_norms**2

                if self.update_count <= 68:
                    opt_flag = "SGD"
                else:
                    target_loss_reduction = lgn**2
            elif "iEFlg2pt" in opt_flag: # iEF 0 for p > p_thresh samples
                p_thresh = float(opt_flag[8:])
                p_filter = (target_prob < p_thresh).to(torch.float)
                target_loss_reduction = lgn**2 * p_filter
            elif "iEFlg" in opt_flag:
                iEF_level = int(opt_flag[5:])
                # use logit_grad_norm to scale updates
                target_loss_reduction = lgn**iEF_level
            elif "iEFp" in opt_flag:
                target_loss_reduction = (1 - target_prob) / (target_prob + 1e-4)
            elif "iEFnewton" in opt_flag:
                target_loss_reduction = newton_lr
            elif "iEFloss" in opt_flag:
                target_loss_reduction = logp_lr
            elif "focal" in opt_flag:
                ps = opt_flag[3:].split("focal")
                iEF_level = int(ps[0])
                focal_level = int(ps[1])
                target_loss_reduction = (
                    grad_inputs_norms**iEF_level * (1 - target_prob) ** focal_level
                )
            else:
                iEF_level = int(opt_flag[3:])
                target_loss_reduction = grad_inputs_norms**iEF_level
            # add weight term weighting when required
            if const_weight:
                print("CONST WEIGHT in effect")
                weight_weight = torch.zeros(1, 1).to(target_loss_reduction.device)
                target_loss_reduction = torch.cat(
                    (target_loss_reduction.unsqueeze(1), weight_weight), dim=0
                ).squeeze()
            # convert to target precision
            target_loss_reduction = target_loss_reduction.to(target_floating_point)
            # convert opt_flag back
            opt_flag = prefix_iEF + opt_flag

        # add weight term in gradient matrix
        param_list = [param.view(-1) for param in self.trainable_parameters_sequence]
        flat_param = torch.cat(param_list, dim=-1)
        if const_weight and "iEF" in opt_flag:
            J = torch.cat((J, flat_param[None, :]), dim=0)
            # raise ValueError("Additional const_weight vector change the meaning of sample_cnt, which would cause issue. \
            #                  If const_weight had to be used, please fix this issue.")
        J_row_cnt = J.shape[0]

        # compute J covariance
        tmp_time_st = time.time()
        C = J @ J.T
        C = C.to(target_floating_point)
        if "ief" in opt_flag.lower():
            self.update_time += time.time() - tmp_time_st

        # compute grad similarity
        diagonal_vector = C.diag() ** 0.5
        cov_norm = (
            diagonal_vector.unsqueeze(dim=-1) @ diagonal_vector.unsqueeze(dim=0)
        )
        cov_sim = C / cov_norm

        # process accumulated batches for debug
        # debug_batch_data = [debug_batch_data[i] for i in sorted_index]

        # for highly similar gradients, preserve only the larger norm one
        remove_idx = []
        SIM_THRESH = 0.99
        for i in range(J_row_cnt):
            for j in range(i+1, J_row_cnt):
                if torch.abs(cov_sim[i, j]) > SIM_THRESH:
                    if j == sample_cnt: # this happens only for the parameter vector when const_weight is activated
                        remove_idx.append(i) # must keep the weight vector
                    else: # keep the larger norm grad
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
            J_row_cnt = len(grad_idx)
            if target_loss_reduction is not None:
                target_loss_reduction = target_loss_reduction[grad_idx]
            # remove the last idx from grad_idx, since it is const_weight term
            if const_weight:
                grad_idx = grad_idx[:-1]
            sample_cnt = len(grad_idx)
            grad_inputs_norms = grad_inputs_norms[grad_idx]
            lgn = lgn[grad_idx]
            newton_lr = newton_lr[grad_idx]
            logp_lr = logp_lr[grad_idx]
            target_prob = target_prob[grad_idx]

        # start computing update
        rel_err = 0
        if "SGD" in opt_flag:
            tmp_time_st = time.time()
            if opt_flag == "SGD":
                new_grad = self.clone_grad_list(self.momentum_grad)
            elif opt_flag.startswith("SGDpt"):
                p_thresh = float(opt_flag[5:])
                p_filter = (target_prob < p_thresh).to(torch.float)
                new_J = p_filter.unsqueeze(dim=-1) * J 
                new_grad = self.flat_grad_to_grad_list(new_J.sum(dim=0))
            else:
                if const_weight:
                    raise ValueError("Higher Order SGD is not compatible with const_weight")
                if momentum != 0:
                    raise NotImplementedError(
                        "Momentum for non-standard SGD are not implemented"
                    )
                SGD_level = int(opt_flag[3:])
                new_J = (
                    grad_inputs_norms.unsqueeze(-1).to(J.dtype) ** (SGD_level - 2) * J
                )
                new_grad = self.flat_grad_to_grad_list(new_J.sum(dim=0))
            self.update_time += time.time() - tmp_time_st
        elif opt_flag.lower().startswith("sf"):
            # gradient preconditioned with sampled fisher
            # compute momentum gradient
            if self.momentum_grad is not None and momentum != 0:
                self.momentum_grad = self.add_grad_list(self.momentum_grad, param_grad_list, momentum, 1, out=self.momentum_grad)
            else:
                self.momentum_grad = self.clone_grad_list(param_grad_list)

            # computes basic configs
            MC_TIME = int(opt_flag[2:])
            if damping < 0: # special median damping
                damping = C.diag().median() * -damping
            g = self.grad_list_to_flat_grad(self.momentum_grad).to(torch.float64) # use momentum grad

            # acquire sampled jacobian
            tmp_time_st = time.time()
            J_hat = self.collect_SF_Jacobian().to(torch.float64)
            C_hat_o = J_hat @ J_hat.T
            # apply damping
            I_hat = torch.eye(C_hat_o.shape[0], dtype=C_hat_o.dtype, device=C_hat_o.device)
            C_hat = C_hat_o + MC_TIME * damping * I_hat
            J_hat_g = J_hat @ g
            C_hat_i_J_hat_g = torch.linalg.solve(C_hat, J_hat_g)
            J_hat_T_C_hat_i_J_hat_g = J_hat.T @ C_hat_i_J_hat_g
            grad_flat_0 = 1/damping*(g - J_hat_T_C_hat_i_J_hat_g)
            g0_time = time.time() - tmp_time_st
            
            # generate update
            new_grad = self.flat_grad_to_grad_list(grad_flat_0)
        elif "iEF" in opt_flag:
            # advanced damping method
            tmp_time_st = time.time()
            damping_matrix = torch.diag(torch.ones_like(target_loss_reduction)) # matching diagonal damping matrix
            if damping == 1: # when damping == 1, only the diagonal of C is kept
                damped_C = torch.diag(C.diag())
            elif damping > 0:
                tmp_damping = damping
                damped_C = C + damping * damping_matrix
            elif damping < 0: # special median damping
                damping = C.diag().max() * -damping
                damped_C = C + damping * damping_matrix


            target_grad_weighting = torch.linalg.solve(damped_C, target_loss_reduction)
            self.update_time += time.time() - tmp_time_st

            # compute relative error
            rel_err = (
                torch.sqrt(((target_grad_weighting @ C - target_loss_reduction) ** 2).mean())
                / target_loss_reduction.mean()
            )

            # compute new gradient
            target_grad_weighting = target_grad_weighting.to(torch.float32)
            grad_flat = target_grad_weighting @ J
            new_grad = self.flat_grad_to_grad_list(grad_flat)

            # apply simple momentum
            if momentum != 0:
                if self.iEF_prev_grad is not None:
                    self.add_grad_list(self.iEF_prev_grad, new_grad, momentum, 1, out=new_grad)
                self.iEF_prev_grad = self.clone_grad_list(new_grad)

            # get update time
            self.update_time += time.time() - tmp_time_st
        else:
            raise ValueError("Unknown opt_flag")

        # compute improved grad's effectiveness
        (self.stats["new_grad_effectivenss"], _, _) = cosine_similarity(
            self.momentum_grad, new_grad
        )
        
        # do adaptive learning 
        if adapt_lr:
            if adapt_lr and line_search:
                raise "Cannot use Adaptive Learning Rate with Line SEARCH!!!!"
            elif clip_norm is not None:
                raise "Cannot use Clip Norm with Adaptive Learning Rate"
            else:
                lr = lr / self.stats["new_grad_effectivenss"]

        # update this grad to parameter
        tmp_time_st = time.time()
        cur_update = self.clone_grad_list(new_grad)
        if norm_update:
            if clip_norm is not None:
                raise "Cannot use Clip Norm with Norm Update"
            else:
                cur_update_norm = self.norm_grad_list(cur_update)
                self.add_grad_list(cur_update, None, alpha=1/(cur_update_norm+1e-12), out=cur_update)
                # cur_update = cur_update / (cur_update_norm + 1e-12)
        elif clip_norm is not None:
            cur_update_norm = self.norm_grad_list(cur_update)
            if cur_update_norm * lr > clip_norm:
                lr = clip_norm / cur_update_norm
        cur_update = self.add_grad_list(cur_update, alpha=-lr, out=cur_update)
        
        # if required to linesearch do line search
        for param, cu in zip(self.trainable_parameters_sequence, cur_update):
            param.data.add_(cu)
        self.update_time += time.time() - tmp_time_st

        # keep normalise weight
        if norm_weight:
            cur_weight_norm = self.norm_grad_list(self.trainable_parameters_sequence)
            for param in self.trainable_parameters_sequence:
                param.data.mul_(self.init_weight_norm / cur_weight_norm)

        # compute stats
        # record iEF error
        self.stats["rel_err"] = rel_err
        self.stats["max_samples"] = sample_cnt
        # compute largest / smallest 10% samples' gradients' ratio
        try:
            portion_cnt = int(sample_cnt_old * 0.1)
        except:
            portion_cnt = int(sample_cnt * 0.1)
        try:
            grad_inputs_norms_sorted, _ = torch.sort(grad_inputs_norms_old)
        except:
            grad_inputs_norms_sorted, _ = torch.sort(grad_inputs_norms)
        min_grad_norm = grad_inputs_norms_sorted[:portion_cnt].mean()
        max_grad_norm = grad_inputs_norms_sorted[-portion_cnt:].mean()
        self.stats["sample_grad_std/mean"] = (
            grad_inputs_norms_sorted.std() / grad_inputs_norms_sorted.mean()
        )
        self.stats["sample_grad_10%max/min"] = max_grad_norm / min_grad_norm
        self.stats["sample_grad_max"] = grad_inputs_norms_sorted[-1]
        # compute largest / smallest 10% samples' gradients' ratio
        try:
            logits_grad_norms_sorted, _ = torch.sort(lgn_old)
        except:
            logits_grad_norms_sorted, _ = torch.sort(lgn)
        min_grad_norm = logits_grad_norms_sorted[:portion_cnt].mean()
        max_grad_norm = logits_grad_norms_sorted[-portion_cnt:].mean()
        self.stats["logits_grad_10%max/min"] = max_grad_norm / min_grad_norm
        # compute logits grad / samples grad ratio
        # compute average cosine similarities of sample gradients in a batch
        if const_weight:  # remove weight term
            C = C[:sample_cnt, :sample_cnt]
        cov_norm = (
            grad_inputs_norms.unsqueeze(dim=-1) @ grad_inputs_norms.unsqueeze(dim=0)
        )
        cov_sim = C / cov_norm
        self.stats["batch_grad_sim"] = (cov_sim[1:, 1:].sum() - sample_cnt + 1) / (
            (sample_cnt - 1) ** 2 - sample_cnt + 1
        )
        # compute consecutive updates' similarity
        (
            self.stats["update_sim"],
            _,
            self.stats["update_norm"],
        ) = cosine_similarity(self.prev_update, cur_update)
        (
            self.stats["original_grad_sim"],
            _,
            self.stats["original_grad_norm"],
        ) = cosine_similarity(self.prev_old_grad, param_grad_list)
        if opt_flag != 3:
            (
                self.stats["new_grad_sim"],
                _,
                self.stats["new_grad_norm"],
            ) = cosine_similarity(self.prev_new_grad, new_grad)
        else:
            self.stats["new_grad_sim"], self.stats["new_grad_norm"] = (
                self.stats["original_grad_sim"],
                self.stats["original_grad_norm"],
            )
        # compute largest / smallest 10% samples' gradients' ratio
        try:
            p_sorted, _ = torch.sort(p_lr_old)
        except:
            p_sorted, _ = torch.sort(target_prob)
        min_p = p_sorted[:portion_cnt].mean()
        max_p = p_sorted[-portion_cnt:].mean()
        self.stats["prob_10%max/min"] = (max_p / min_p).item()
        self.stats["prob_mean"] = p_sorted.mean().item()
        self.stats["prob_std/mean"] = (p_sorted.std() / p_sorted.mean()).item()
        # compute weight norm
        self.stats["weight_norm"] = self.norm_grad_list(self.trainable_parameters_sequence)
        self.stats["converged_count"] = (target_prob >= 0.5).sum()
        # time stats
        self.stats["update_time"] = self.update_time
        self.update_time = 0

        # save current grad_inputs_matrix to the prev
        self.prev_grad_inputs_matrix = J
        self.prev_old_grad = self.clone_grad_list(param_grad_list)
        self.prev_new_grad = self.clone_grad_list(new_grad)
        self.prev_update = self.clone_grad_list(cur_update)