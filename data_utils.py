import torch
from torch.utils.data import Dataset
import os

lan_dict = {"de":"de_DE","en":"en_XX","nl":"nl_XX",
            "es":"es_XX","it":"it_IT","fr":"fr_XX",
            "pl":"pl_PL","sl":"sl_SI","ru":"ru_RU",
            "lt":"lt_LT","lv":"lv_LV",
            "zh":"zh_CN","zh_cn":"zh_CN","th":"th_TH",
            "ar":"ar_AR","he":"he_IL",
            "fi":"fi_FI","et":"et_EE"}

lang_group_ep = {"de":0,"en":0,"nl":0,"fr":1,"it":1,"es":1,"pl":2,"sl":2,"lt":3,"lv":3}
lang_group_ted = {"zh_cn":0,"en":0,"th":0,"ar":1,"he":1,"fi":2,"et":2,"ru":3,"sl":3}

num_instances_ted = {
    "zh_cn-en":{"train":10000,"dev":3000,"test":3000},
    "th-en":{"train":5000,"dev":1500,"test":1500},
    "ar-en":{"train":10000,"dev":3000,"test":3000},
    "he-en":{"train":2000,"dev":600,"test":600},
    "fi-en":{"train":2000,"dev":600,"test":600},
    "et-en":{"train":2000,"dev":600,"test":600},
    "ru-en":{"train":10000,"dev":3000,"test":3000},
    "sl-en":{"train":2000,"dev":600,"test":600},
}


def load_lang_pairs(path):
    file = open(path,"r",encoding="utf8")
    lang_pairs = []
    for line in file.readlines():
        lang_pair = line.strip()
        lang_pairs.append(lang_pair)
    return lang_pairs


class TransDataset(Dataset):
    def __init__(self,src_data,trg_data,src_lang,trg_lang) -> None:
        super(TransDataset,self).__init__()
        self.src_lang, self.trg_lang = src_lang, trg_lang
        self.examples = []
        for s,t in zip(src_data,trg_data):
            self.examples.append((torch.tensor(s, dtype=torch.long), torch.tensor(t, dtype=torch.long)))
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, index):
        return self.examples[index]

def check_data(line,lang):
    if lang=="zh_cn" or lang=="zh" or lang=="th":
        if len(line)>1:
            return True
        else:
            return False
    else:
        if len(line.split())>1:
            return True
        else:
            return False

def load_and_split_ep_data(args,model_dict):
    lang_pair_path = args.lang_pair_dir + f"{args.model}_{args.dataset}_{args.mode}.txt"
    lang_pairs = load_lang_pairs(lang_pair_path)
    datasets = {"train":[],"dev":[],"test":[]}
    uniform_datasize = {"train":5000,"dev":1500,"test":1500}
    tokenizer = model_dict[args.model][-1].from_pretrained(args.pretrain_path)
    for lang_pair in lang_pairs:
        src_name, trg_name = lang_pair.split("-")
        base_path = "./data/Europarl/processed_data/"+src_name+"-"+trg_name+"/"
        if not os.path.exists(base_path):
            base_path = "./data/Europarl/processed_data/"+trg_name+"-"+src_name+"/"
        for type in ["train","dev","test"]:
            src_data = open(base_path+src_name+"."+type,"r",encoding="utf8").readlines()
            trg_data = open(base_path+trg_name+"."+type,"r",encoding="utf8").readlines()
            if not args.uniform:
                num_instances = int(len(src_data)/100)
            else:
                num_instances = uniform_datasize[type]
            if type=="train":
                bs = args.batch_size*args.local_steps
                max_len = (num_instances // bs)*bs
                num_instances = max_len
            src_examples = []
            trg_examples = []
            if args.model == "mbart":
                src_lang = lan_dict[src_name]
                trg_lang = lan_dict[trg_name]
            else:
                src_lang, trg_lang = src_name, trg_name
            for _ in range(len(src_data)):
                src_line_data = src_data[_].strip()
                trg_line_data = trg_data[_].strip()
                if (not check_data(src_line_data,src_name)) or (not check_data(trg_line_data,trg_name)):
                    continue
                src_examples.append(src_line_data)
                trg_examples.append(trg_line_data)
                if len(src_examples)>=num_instances:
                    break
            tokenizer.src_lang = src_lang
            src_inputs = tokenizer(src_examples, add_special_tokens=True, truncation=True, max_length=args.max_seq_length)["input_ids"]
            tokenizer.src_lang = trg_lang
            trg_inputs = tokenizer(trg_examples, add_special_tokens=True, truncation=True, max_length=args.max_seq_length)["input_ids"]
            dataset = TransDataset(src_inputs, trg_inputs,src_name,trg_name)
            datasets[type].append(dataset)
    return datasets["train"],datasets["dev"],datasets["test"]

def load_ep_data(args,model_dict):
    lang_pair_path = args.lang_pair_dir + f"{args.model}_{args.dataset}_{args.mode}.txt"
    lang_pairs = load_lang_pairs(lang_pair_path)
    src_total_inputs = {"train":[],"dev":[],"test":[]}
    trg_total_inputs = {"train":[],"dev":[],"test":[]}
    tokenizer = model_dict[args.model][-1].from_pretrained(args.pretrain_path)
    for lang_pair in lang_pairs:
        src_name, trg_name = lang_pair.split("-")
        base_path = "./data/Europarl/processed_data/"+src_name+"-"+trg_name+"/"
        if not os.path.exists(base_path):
            base_path = "./data/Europarl/processed_data/"+trg_name+"-"+src_name+"/"
        for type in ["train","dev","test"]:
            src_data = open(base_path+src_name+"."+type,"r",encoding="utf8").readlines()
            trg_data = open(base_path+trg_name+"."+type,"r",encoding="utf8").readlines()
            num_instances = int(len(src_data)/100)
            if type=="train":
                bs = args.batch_size*args.local_steps
                max_len = (num_instances // bs)*bs
                num_instances = max_len
            src_examples = []
            trg_examples = []
            if args.model == "mbart":
                src_lang = lan_dict[src_name]
                trg_lang = lan_dict[trg_name]
            else:
                src_lang, trg_lang = src_name, trg_name
            for _ in range(len(src_data)):
                src_line_data = src_data[_].strip()
                trg_line_data = trg_data[_].strip()
                if (not check_data(src_line_data,src_name)) or (not check_data(trg_line_data,trg_name)):
                    continue
                src_examples.append(src_line_data)
                trg_examples.append(trg_line_data)
                if len(src_examples)>=num_instances:
                    break
            tokenizer.src_lang = src_lang
            src_inputs = tokenizer(src_examples, add_special_tokens=True, truncation=True, max_length=args.max_seq_length)["input_ids"]
            tokenizer.src_lang = trg_lang
            trg_inputs = tokenizer(trg_examples, add_special_tokens=True, truncation=True, max_length=args.max_seq_length)["input_ids"]
            src_total_inputs[type].extend(src_inputs)
            trg_total_inputs[type].extend(trg_inputs)
    train_dataset = TransDataset(src_total_inputs["train"], trg_total_inputs["train"],"centralized","centralized")
    dev_dataset = TransDataset(src_total_inputs["dev"], trg_total_inputs["dev"],"centralized","centralized")
    test_dataset = TransDataset(src_total_inputs["test"], trg_total_inputs["test"],"centralized","centralized")
    return train_dataset,dev_dataset,test_dataset

def load_and_split_ted_data(args,model_dict):
    lang_pair_path = args.lang_pair_dir + f"{args.model}_{args.dataset}_{args.mode}.txt"
    lang_pairs = load_lang_pairs(lang_pair_path)
    datasets = {"train":[],"dev":[],"test":[]}
    uniform_datasize = {"train":5000,"dev":1500,"test":1500}
    tokenizer = model_dict[args.model][-1].from_pretrained(args.pretrain_path)
    for lang_pair in lang_pairs:
        src_name, trg_name = lang_pair.split("-")
        base_path = "./data/TED2020/processed_data/"+src_name+"-"+trg_name+"/"
        if not os.path.exists(base_path):
            base_path = "./data/TED2020/processed_data/"+trg_name+"-"+src_name+"/"
        for type in ["train","dev","test"]:
            src_data = open(base_path+src_name+"."+type,"r",encoding="utf8").readlines()
            trg_data = open(base_path+trg_name+"."+type,"r",encoding="utf8").readlines()
            if not args.uniform:
                num_instances = num_instances_ted[lang_pair][type]
            else:
                num_instances = uniform_datasize[type]
            if type=="train":
                bs = args.batch_size*args.local_steps
                max_len = (num_instances // bs)*bs
                num_instances = max_len
            src_examples = []
            trg_examples = []
            if args.model == "mbart":
                src_lang = lan_dict[src_name]
                trg_lang = lan_dict[trg_name]
            else:
                src_lang, trg_lang = src_name, trg_name
            for _ in range(len(src_data)):
                src_line_data = src_data[_].strip()
                trg_line_data = trg_data[_].strip()
                if (not check_data(src_line_data,src_name)) or (not check_data(trg_line_data,trg_name)):
                    continue
                src_examples.append(src_line_data)
                trg_examples.append(trg_line_data)
                if len(src_examples)>=num_instances:
                    break
            tokenizer.src_lang = src_lang
            src_inputs = tokenizer(src_examples, add_special_tokens=True, truncation=True, max_length=args.max_seq_length)["input_ids"]
            tokenizer.src_lang = trg_lang
            trg_inputs = tokenizer(trg_examples, add_special_tokens=True, truncation=True, max_length=args.max_seq_length)["input_ids"]
            dataset = TransDataset(src_inputs, trg_inputs,src_name,trg_name)
            datasets[type].append(dataset)
    return datasets["train"],datasets["dev"],datasets["test"]

def load_ted_data(args,model_dict):
    lang_pair_path = args.lang_pair_dir + f"{args.model}_{args.dataset}_{args.mode}.txt"
    lang_pairs = load_lang_pairs(lang_pair_path)
    src_total_inputs = {"train":[],"dev":[],"test":[]}
    trg_total_inputs = {"train":[],"dev":[],"test":[]}
    tokenizer = model_dict[args.model][-1].from_pretrained(args.pretrain_path)
    for lang_pair in lang_pairs:
        src_name, trg_name = lang_pair.split("-")
        base_path = "./data/TED2020/processed_data/"+src_name+"-"+trg_name+"/"
        if not os.path.exists(base_path):
            base_path = "./data/TED2020/processed_data/"+trg_name+"-"+src_name+"/"
        for type in ["train","dev","test"]:
            src_data = open(base_path+src_name+"."+type,"r",encoding="utf8").readlines()
            trg_data = open(base_path+trg_name+"."+type,"r",encoding="utf8").readlines()
            num_instances = num_instances_ted[lang_pair][type]
            if type=="train":
                bs = args.batch_size*args.local_steps
                max_len = (num_instances // bs)*bs
                num_instances = max_len
            src_examples = []
            trg_examples = []
            if args.model == "mbart":
                src_lang = lan_dict[src_name]
                trg_lang = lan_dict[trg_name]
            else:
                src_lang, trg_lang = src_name, trg_name
            for _ in range(len(src_data)):
                src_line_data = src_data[_].strip()
                trg_line_data = trg_data[_].strip()
                if (not check_data(src_line_data,src_name)) or (not check_data(trg_line_data,trg_name)):
                    continue
                src_examples.append(src_line_data)
                trg_examples.append(trg_line_data)
                if len(src_examples)>=num_instances:
                    break
            tokenizer.src_lang = src_lang
            src_inputs = tokenizer(src_examples, add_special_tokens=True, truncation=True, max_length=args.max_seq_length)["input_ids"]
            tokenizer.src_lang = trg_lang
            trg_inputs = tokenizer(trg_examples, add_special_tokens=True, truncation=True, max_length=args.max_seq_length)["input_ids"]
            src_total_inputs[type].extend(src_inputs)
            trg_total_inputs[type].extend(trg_inputs)
    train_dataset = TransDataset(src_total_inputs["train"], trg_total_inputs["train"],"centralized","centralized")
    dev_dataset = TransDataset(src_total_inputs["dev"], trg_total_inputs["dev"],"centralized","centralized")
    test_dataset = TransDataset(src_total_inputs["test"], trg_total_inputs["test"],"centralized","centralized")
    return train_dataset,dev_dataset,test_dataset