import os
import torch
import argparse
from rm import *
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import (
    SAFE_WEIGHTS_NAME,
)

def fix_valuehead(
    model, rm_ckpt_dir: str, rm_save_dir: str, V_HEAD_WEIGHTS_NAME: str = "value_head.bin"
) -> None:

    path_to_checkpoint = os.path.join(rm_ckpt_dir, SAFE_WEIGHTS_NAME)
    state_dict = load_file(path_to_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(
        rm_ckpt_dir, 
        trust_remote_code=True, 
        use_fast=True,
        padding_side="right",
        split_special_tokens=False,
    )
    model.pretrained_model.resize_token_embeddings(len(tokenizer))
    decoder_state_dict = {}
    v_head_state_dict = {}
    for name, param in state_dict.items():
        if name.startswith("v_head."):
            v_head_state_dict[name] = param
        else:
            decoder_state_dict[name.replace("pretrained_model.", "", 1)] = param

    model.pretrained_model.save_pretrained(
        rm_save_dir, state_dict=decoder_state_dict or None
    )
    tokenizer.save_pretrained(rm_save_dir)
    torch.save(v_head_state_dict, os.path.join(rm_save_dir, V_HEAD_WEIGHTS_NAME))
    #os.remove(path_to_checkpoint)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--sft_model_path", type=str, default="sft_model_path")
    args.add_argument("--rm_ckpt_path", type=str, default="rm_ckpt_path")
    args.add_argument("--rm_save_path", type=str, default="rm_save_path")
    args = args.parse_args()
    
    model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path, 
        trust_remote_code=True,
        #torch_dtype=torch.bfloat16,
        use_cache=False,
    )
    model = RewardModelWithValueHead(pretrained_model=model)
    fix_valuehead(model, args.rm_ckpt_path, args.rm_save_path)