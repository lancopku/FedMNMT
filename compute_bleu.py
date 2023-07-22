from sacrebleu.metrics import BLEU
from data_utils import load_lang_pairs
from tqdm import tqdm
import numpy as np
import argparse
import os

def str2bool(arg):
    if arg == 'True':
        return True
    else:
        return False

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",type=str,default="mbart")
    parser.add_argument("--dataset", type=str, default="europarl")
    parser.add_argument("--mode",type=str, default="m2m")
    parser.add_argument("--use_adapter",type=str2bool, default=True)
    parser.add_argument("--share",type=str, default="shareAll")
    parser.add_argument("--uniform",type=str2bool, default=False)
    parser.add_argument("--evaluation_dir",type=str,default="./evaluation")
    parser.add_argument("--lang_pair_dir",type=str, default="./exp_lang_pairs")
    args = parser.parse_args()
    return args


args = get_args()
adapter_str = "a" if args.use_adapter else "wa"
if args.uniform:
    adapter_str += "_uniform"
base_dir = f"{args.evaluation_dir}/{args.model}/{args.dataset}/{args.mode}_{adapter_str}_{args.share}/"
print(base_dir)
lang_pair_path = f"{args.lang_pair_dir}/{args.model}_{args.dataset}_{args.mode}.txt"
lang_pairs = load_lang_pairs(lang_pair_path)
raw_list_dirs = os.listdir(base_dir)
list_dirs = []
for name in raw_list_dirs:
    if "2022" in name or "2023" in name or "2024" in name:
        list_dirs.append(name)
print(list_dirs)
avg_bleu_score = {}
for dir_name in list_dirs:
    eva_dir = base_dir + dir_name
    num_clients = len(lang_pairs)
    ground_truth = [[] for _ in range(num_clients)]
    prediction = [[] for _ in range(num_clients)]
    macro_bleu_score, micro_bleu_score = 0, 0
    for i in range(num_clients):
        src_lang, trg_lang = lang_pairs[i].split("-")
        g_path = eva_dir + f"/client{i}_ground_truth_{src_lang}_{trg_lang}.txt"
        p_path = eva_dir + f"/client{i}_prediction_{src_lang}_{trg_lang}.txt"
        g = open(g_path,"r",encoding="utf8")
        p = open(p_path,"r",encoding="utf8")
        for line in g.readlines():
            ground_truth[i].append(line.strip())
        for line in p.readlines():
            prediction[i].append(line.strip())
    total_len = 0
    for i in range(num_clients):  
        gt, pd = ground_truth[i], prediction[i]
        src_lang, trg_lang = lang_pairs[i].split("-")
        bleu = BLEU(effective_order=True)
        lang_pair_bleu = bleu.corpus_score(pd,[gt]).score
        if lang_pairs[i] in avg_bleu_score:
            avg_bleu_score[lang_pairs[i]] += lang_pair_bleu
        else:
            avg_bleu_score[lang_pairs[i]] = lang_pair_bleu
        macro_bleu_score += lang_pair_bleu
        micro_bleu_score += lang_pair_bleu*len(gt)
        total_len += len(gt)
    macro_bleu_score /= len(ground_truth)
    micro_bleu_score /= total_len
    if "macro" in avg_bleu_score:
        avg_bleu_score["macro"] += macro_bleu_score
    else:
        avg_bleu_score["macro"] = macro_bleu_score
    if "micro" in avg_bleu_score:
        avg_bleu_score["micro"] += micro_bleu_score
    else:
        avg_bleu_score["micro"] = micro_bleu_score
for k, v in avg_bleu_score.items():
    avg_bleu_score[k] = v / len(list_dirs)
print("Average")
for lang_pair in lang_pairs:
    print("%.2f"%(avg_bleu_score[lang_pair]),end=" & ")
print("%.2f"%(avg_bleu_score["macro"]),end=" & ")
print("%.2f"%(avg_bleu_score["micro"]))