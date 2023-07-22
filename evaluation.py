import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import argparse
from copy import deepcopy
from transformers import MBartForConditionalGeneration,MBart50Tokenizer
from mbart_adapter_model import MBartAdapterForConditionalGeneration
import os
from data_utils import load_and_split_ep_data, load_and_split_ted_data
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random




model_dict = {"mbart":(MBartForConditionalGeneration,MBartAdapterForConditionalGeneration,MBart50Tokenizer)}

def str2bool(arg):
    if arg == 'True':
        return True
    else:
        return False

def str2list(arg):
    tmp = arg.split(",")
    result = [int(_) for _ in tmp]
    return result

def eva_collate_function(batch):
    batched_input_tensors = pad_sequence([s for (s, t) in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    batched_label_tensors = pad_sequence([t for (s, t) in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    attn_mask = torch.ones_like(batched_input_tensors)
    is_padding = (batched_input_tensors == tokenizer.pad_token_id)
    attn_mask[is_padding] = 0
    output_batch = {
        "attention_mask": attn_mask,
        "labels": batched_label_tensors,
        "input_ids": batched_input_tensors
    }
    return output_batch

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,default=2022)
    parser.add_argument("--model",type=str,default="mbart")
    parser.add_argument("--dataset",type=str,default='europarl')
    parser.add_argument("--max_seq_length",type=int, default=256)
    parser.add_argument("--batch_size",type=int, default=8)
    parser.add_argument("--local_steps",type=int, default=16)
    parser.add_argument("--eva_batch_size",type=int, default=32)
    parser.add_argument("--use_adapter",type=str2bool, default=True)
    parser.add_argument("--share",type=str, default="shareAll")
    parser.add_argument("--uniform",type=str2bool, default=False)
    parser.add_argument("--mode",type=str, default="m2m")
    parser.add_argument("--log_dir",type=str, default="./logs")
    parser.add_argument("--ckpt_base_dir",type=str, default="./models")
    parser.add_argument("--evaluation_dir",type=str, default="./evaluation")
    parser.add_argument("--device",type=str, default="cuda:0")
    parser.add_argument("--pretrain_path",type=str, default="./mbart-large-50-many-to-many-mmt")
    parser.add_argument("--lang_pair_dir",type=str, default="./exp_lang_pairs/")
    
    args = parser.parse_args()
    return args

def evaluate(datasets,model):
    adapter_str = "a" if args.use_adapter else "wa"
    if args.uniform:
        adapter_str += "_uniform"
    log_path = args.log_dir + "/" + args.model + "/" + args.dataset +"/" + str(args.seed) + "/" +args.mode+"_"+adapter_str+"_"+args.share+".log"
    print(log_path)
    ckpt_dir = args.ckpt_base_dir+ "/" + args.model + "/" + args.dataset+"/"+ args.mode + "_" + adapter_str+"_"+args.share+"/"+str(args.seed)+"/"
    evaluation_dir = args.evaluation_dir + "/" + args.model + "/" + args.dataset+"/"+args.mode + "_" +adapter_str+"_"+args.share+"/"+str(args.seed)+"/"
    print(ckpt_dir)
    if not os.path.exists(evaluation_dir):
        os.makedirs(evaluation_dir)
    for i in range(num_clients):
        dataset = datasets[i]
        src_lang, trg_lang = dataset.src_lang, dataset.trg_lang
        f_ground = open(evaluation_dir+f"client{i}_ground_truth_{src_lang}_{trg_lang}.txt","w",encoding="utf-8")
        f_prediction = open(evaluation_dir+f"client{i}_prediction_{src_lang}_{trg_lang}.txt","w",encoding="utf-8")
        print(f"Client {i}:")
        
        if args.share == "centralized":
            model_dict = torch.load(ckpt_dir+f"best_model.pt",map_location="cpu")
            model.load_state_dict(model_dict)
        else:
            model_dict = torch.load(ckpt_dir+f"client{i}_best_model.pt",map_location="cpu")
            model.load_state_dict(model_dict)

        model = model.to(device)
        model.eval()
        dataloader = DataLoader(dataset,batch_size=args.eva_batch_size,shuffle=False,collate_fn=eva_collate_function)
        ground_truth = []
        prediction = []
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            label_ids = batch["labels"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            x = {"input_ids": input_ids, "attention_mask": attn_mask}
            output = model.generate(**x, forced_bos_token_id=batch["labels"][:, 0][0].item(),max_length=args.max_seq_length)
            prediction.extend(tokenizer.batch_decode(output, skip_special_tokens=True))
            ground_truth.extend(tokenizer.batch_decode(label_ids, skip_special_tokens=True))
        for g,p in zip(ground_truth,prediction):
            f_ground.write(g+"\n")
            f_prediction.write(p+"\n")
        model = model.cpu()

if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    print(args.seed)
    device = torch.device(args.device)
    model_class, adapter_model_class, tokenizer_class = model_dict[args.model]
    tokenizer = tokenizer_class.from_pretrained(args.pretrain_path)

    if args.dataset == 'europarl':
        train_datasets, dev_datasets, test_datasets = load_and_split_ep_data(args,model_dict)
    elif args.dataset == 'ted2020':
        train_datasets, dev_datasets, test_datasets = load_and_split_ted_data(args,model_dict)

    num_clients = len(train_datasets)
    if not args.use_adapter:
        model = model_class.from_pretrained(args.pretrain_path)
    else:
        model = adapter_model_class.from_pretrained(args.pretrain_path)

    evaluate(test_datasets,model)