import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from transformers import MBartForConditionalGeneration,MBart50Tokenizer
from mbart_adapter_model import MBartAdapterForConditionalGeneration
from torch.nn.utils.rnn import pad_sequence
import argparse
from tqdm import tqdm
from copy import deepcopy
from data_utils import load_and_split_ep_data,load_and_split_ted_data,lang_group_ep,lang_group_ted
import numpy as np
import random
import os

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
    parser.add_argument("--dataset", type=str, default="ted2020")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--local_lr",type=float, default=1e-3)
    parser.add_argument("--local_steps",type=int, default=16)
    parser.add_argument("--max_rounds",type=int, default=5)
    parser.add_argument("--mode",type=str, default="m2en")
    parser.add_argument("--max_seq_length",type=int, default=256)
    parser.add_argument("--use_adapter",type=str2bool, default=True)
    parser.add_argument("--share",type=str, default="shareAll")
    parser.add_argument("--uniform",type=str2bool, default=False)
    parser.add_argument("--log_dir",type=str, default="./logs")
    parser.add_argument("--ckpt_base_dir",type=str, default="./models")
    parser.add_argument("--device",type=str, default="cuda:0")
    parser.add_argument("--device_ids",type=str2list, default=[0,1,2,3])
    parser.add_argument("--pretrain_path",type=str, default="./mbart-large-50-many-to-many-mmt")
    parser.add_argument("--lang_pair_dir",type=str, default="./exp_lang_pairs/")
    args = parser.parse_args()
    return args

def collate_function(batch):
    batched_input_tensors = pad_sequence([s for (s, t) in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    batched_label_tensors = pad_sequence([t for (s, t) in batch], batch_first=True, padding_value=-100)
    attn_mask = torch.ones_like(batched_input_tensors)
    is_padding = (batched_input_tensors == tokenizer.pad_token_id)
    attn_mask[is_padding] = 0
    output_batch = {
        "attention_mask": attn_mask,
        "labels": batched_label_tensors,
        "input_ids": batched_input_tensors
    }
    return output_batch

def validate():
    total_loss = 0
    total_len = 0
    for i in range(num_clients):
        print(f"Client {i}:")
        dev_dataset = dev_datasets[i]
        total_len += len(dev_dataset)
        model = deepcopy(models[i])
        model = model.to(device)
        model.eval()
        devloader = DataLoader(dev_dataset,batch_size=args.batch_size,shuffle=False,collate_fn=collate_function)
        for batch in tqdm(devloader):
            input_ids = batch["input_ids"].to(device)
            label_ids = batch["labels"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            x = {"input_ids": input_ids, "labels": label_ids,"attention_mask": attn_mask}
            with torch.no_grad():
                total_loss += model(**x).loss.item()*input_ids.size(0)
        model = model.cpu()
        models[i] = deepcopy(model)
    total_loss /= total_len
    return total_loss

def get_share_list():
    encoder_share_list, decoder_share_list = [], []
    if args.dataset == "europarl":
        lang_group = lang_group_ep
    elif args.dataset == "ted2020":
        lang_group = lang_group_ted
    src_lang_list, trg_lang_list = [],[]
    for dataset in train_datasets:
        src_lang_list.append(dataset.src_lang)
        trg_lang_list.append(dataset.trg_lang)
    for client_id, src_name in enumerate(src_lang_list):
        group_id = lang_group[src_name]
        while len(encoder_share_list)<=group_id:
            encoder_share_list.append([])
        encoder_share_list[group_id].append(client_id)
    for client_id, trg_name in enumerate(trg_lang_list):
        group_id = lang_group[trg_name]
        while len(decoder_share_list)<=group_id:
            decoder_share_list.append([])
        decoder_share_list[group_id].append(client_id)
    return encoder_share_list, decoder_share_list

def get_random_share_list(family_num):
    share_list = [[] for _ in range(family_num)]
    num_lang_each_family = num_clients // family_num
    clients_index_list = [_ for _ in range(num_clients)]
    random.shuffle(clients_index_list)
    for family_id in range(family_num):
        start_index = num_lang_each_family*family_id
        end_index = start_index+num_lang_each_family
        for client_id in clients_index_list[start_index:end_index]:
            share_list[family_id].append(client_id)
    for client_id in clients_index_list[end_index:]:
        share_list[-1].append(client_id)  
    return share_list

def get_fix_share_list(args):
    share_list_path = "./share_list/" + f"{args.model}_{args.dataset}_{args.mode}.json"
    import json
    share_list = json.load(open(share_list_path,"r"))
    encoder_share_list, decoder_share_list = share_list["encoder"], share_list["decoder"]
    return encoder_share_list, decoder_share_list

def train():
    file = open(log_path,"w",encoding="utf8")

    if args.share=="shareAll":
        encoder_share_list, decoder_share_list = [[i for i in range(num_clients)]],[[i for i in range(num_clients)]]
    if args.share=="shareLang":
        encoder_share_list, decoder_share_list = get_share_list()
    elif args.share=="random":
        tmp_encoder_share_list, tmp_decoder_share_list = get_share_list()
        encoder_family_num, decoder_family_num = len(tmp_encoder_share_list), len(tmp_decoder_share_list)
        encoder_share_list = get_random_share_list(encoder_family_num)
        decoder_share_list = get_random_share_list(decoder_family_num)
    elif args.share=="shareFix":
        encoder_share_list,decoder_share_list = get_fix_share_list(args)
    try:
        print("Encoder share:",encoder_share_list)
        print("Decoder share:",decoder_share_list)
    except:
        pass
    
    best_loss, best_round = 1e10, 0
    for r in range(1,args.max_rounds+1):
        #train
        print(f"Round {r}")
        losses = []
        for i in range(num_clients):
            print(f"Client {i}: {train_datasets[i].src_lang}->{train_datasets[i].trg_lang}")
            model = deepcopy(models[i])
            model = nn.DataParallel(model.to(device),device_ids=args.device_ids)
            model.train()
            if args.use_adapter:
                optimizer = Adam(filter(lambda x:x.requires_grad, model.parameters()),lr=args.local_lr)
            else:
                optimizer = Adam(model.parameters(),lr=args.local_lr)
            client_loss, acc_step = 0, 0
            update_count = 0
            optimizer.zero_grad()
            for batch in tqdm(train_loaders[i]):
                input_ids = batch["input_ids"].to(device)
                label_ids = batch["labels"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                x = {"input_ids": input_ids, "labels": label_ids, "attention_mask": attn_mask}
                
                loss = torch.mean(model(**x).loss) / args.local_steps
                loss.backward()
                client_loss += loss.item()

                acc_step += 1
                if acc_step == args.local_steps:
                    optimizer.step()
                    acc_step = 0
                    optimizer.zero_grad()
                    update_count += 1
            
            model = model.module.cpu()
            models[i] = deepcopy(model)
            losses.append(client_loss/update_count)
        
        file.write(f"Round {r}\n")
        for i in range(num_clients):
            file.write(f"Client {i}: {train_datasets[i].src_lang}->{train_datasets[i].trg_lang}, {losses[i]}\n")

        #aggregation
        if args.share=="shareAll":
            for key, para in global_model.named_parameters():
                if para.requires_grad:
                    global_model.state_dict()[key].data.zero_()
                    for i in range(num_clients):
                        global_model.state_dict()[key].data.add_(models[i].state_dict()[key].data / num_clients)
            for key, para in global_model.named_parameters():
                if para.requires_grad:
                    for i in range(num_clients):
                        models[i].state_dict()[key].data.copy_(global_model.state_dict()[key].data)
        elif args.share=="shareLang" or args.share=="random" or args.share=="shareFix":
            for ids in encoder_share_list:
                for key, para in global_model.model.encoder.named_parameters():
                    if para.requires_grad:
                        global_model.model.encoder.state_dict()[key].data.zero_()
                        for id in ids:
                            global_model.model.encoder.state_dict()[key].data.add_(models[id].model.encoder.state_dict()[key].data / len(ids))
                for key, para in global_model.model.encoder.named_parameters():
                    if para.requires_grad:
                        for id in ids:
                            models[id].model.encoder.state_dict()[key].data.copy_(global_model.model.encoder.state_dict()[key].data)
            for ids in decoder_share_list:
                for key, para in global_model.model.decoder.named_parameters():
                    if para.requires_grad:
                        global_model.model.decoder.state_dict()[key].data.zero_()
                        for id in ids:
                            global_model.model.decoder.state_dict()[key].data.add_(models[id].model.decoder.state_dict()[key].data / len(ids))
                for key, para in global_model.model.decoder.named_parameters():
                    if para.requires_grad:
                        for id in ids:
                            models[id].model.decoder.state_dict()[key].data.copy_(global_model.model.decoder.state_dict()[key].data)

        # validate
        print("validating......")
        dev_loss = validate()

        if dev_loss < best_loss:
            best_loss = dev_loss
            best_round = r
            for i in tqdm(range(num_clients)):
                torch.save(models[i].state_dict(),ckpt_dir+f"client{i}_best_model.pt")

        print(f"Loss on dev_datasets is {dev_loss}.")
        file.write(f"Loss on dev_datasets is {dev_loss}.\n")
        
        file.write("\n")
        file.flush()

    file.write(f"Best round is round {best_round}.\n")
    file.close()

if __name__ == "__main__":
    args = get_args()
    print(args.seed)
    set_seed(args.seed)
    model_class, adapter_model_class, tokenizer_class = model_dict[args.model]
    tokenizer = tokenizer_class.from_pretrained(args.pretrain_path)
    
    if args.dataset == "europarl":
        train_datasets, dev_datasets, test_datasets = load_and_split_ep_data(args,model_dict)
    elif args.dataset == "ted2020":
        train_datasets, dev_datasets, test_datasets = load_and_split_ted_data(args,model_dict)
    num_clients = len(train_datasets)

    device = torch.device(args.device)
    print(f"Use adapter is {args.use_adapter}")
    print("Length of training datasets is:")
    print([len(train_dataset) for train_dataset in train_datasets])

    if not args.use_adapter:
        models = [model_class.from_pretrained(args.pretrain_path) for _ in range(num_clients)]
    else:
        first_model = adapter_model_class.from_pretrained(args.pretrain_path)
        models = []
        for _ in range(num_clients):
            models.append(deepcopy(first_model))
        for i in range(num_clients):
            for name,para in models[i].named_parameters():
                if "adapter" not in name and "layer_norm" not in name:
                    para.requires_grad = False
    global_model = deepcopy(models[0])
    train_loaders = [DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,collate_fn=collate_function) for train_dataset in train_datasets]
    adapter_str = "a" if args.use_adapter else "wa"
    if args.uniform:
        adapter_str += "_uniform"
    log_dir = args.log_dir + "/" + args.model + "/" + args.dataset +"/" + str(args.seed) + "/"
    ckpt_dir = args.ckpt_base_dir+ "/" + args.model + "/" + args.dataset+"/"+ args.mode + "_" + adapter_str+"_"+args.share+"/" + str(args.seed) + "/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    log_path = log_dir+args.mode+"_"+adapter_str+"_"+args.share+".log"

    train()
