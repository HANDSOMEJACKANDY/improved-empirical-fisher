## Introduction
Instructions for reproducing key experiments of paper [An Improved Empirical Fisher Approximation for Natural Gradient Descent](https://arxiv.org/abs/2406.06420) are provided here.

The code for the implementation for exact Improved Empirical Fisher (iEF) optimiser and our empirical Evaluation Framework is provided here.

Evaluation and Training for two selected setups: GLUE SST2 + Prompt Tuning + T5 and CIFAR100 + LoRA + ViT are provided
To train or evaluate on other tasks, change the corresponding config files accordingly.

## Initialisation
Move to the directory of the repo

Run the following line to create virtual environment ***iEF*** and install all needed packages

> bash ./bash_files/requirements_installation.sh

## Run Experiments:
*Make sure you are registered with wandb.ai to see training curves, also make sure the virtual environment iEF is activated*

1. Run evaluation on SST2 + Prompt Tuning + T5
An example checkpoint for the 11th epoch of SSG2 + Prompt Tuning + T5 trained with Adafactor is provided in ./sample_ckpts/sst2_t5_prompt_tuning_epoch11
run the following line to do an evaluation on several damping factors for this checkpoint
results are outputed in ./sample_ckpts/sst2_t5_prompt_tuning_epoch9/precond_eval, ending with -dmp-{damping factor}.npy

> python3 ./run.py --seed 1337 -m evaluate-preconditioner --expdir ./expdir --config ./configs/t5-prompt-tuning-preconditioner-evaluation.yaml --wandb_project precond_eval_glue --wandb_run_name sst2_t5_pt

2. Run evaluation on CIFAR100 + LoRA + ViT
An example checkpoint for the 9th epoch of SST2 + Prompt Tuning + T5 trained with Adafactor is provided in ./example_ckpts/cifar_vit_lora_e2
run the following line to do an evaluation on several damping factors for this checkpoint
results are outputed in ./example_ckpts/cifar_vit_lora_e2/precond_eval, ending with -dmp-{damping factor}.npy

> python3 ./run.py --seed 1337 -m evaluate-preconditioner-vis --expdir ./expdir --config ./configs/vit-lora-preconditioner-evaluation.yaml --wandb_project precond_eval_cifar --wandb_run_name cifar100_vit_lora

3. Train with Prompt Tuning for SST2 + T5:
run the following lines

> python3 ./run.py --seed 1337 -m train  --expdir ./expdir --config ./configs/t5-prompt-tuning-iEF.yaml --wandb_project train_glue --wandb_run_name sst2_t5_pt_iEF

4. Train with LoRA for CIFAR100 + ViT:
run the following lines

> python3 ./run.py --seed 1337 -m train-vis  --expdir ./expdir --config ./configs/vit-lora-iEF.yaml  --wandb_project train_cifar100 --wandb_run_name cifar100_vit_lora_iEF

