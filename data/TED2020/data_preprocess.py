import random
import os
raw_dir = "./raw_data/"
target_dir = "./processed_data/"
if not os.path.exists(target_dir):
    os.mkdir(target_dir)

random.seed(2022)
dirs = sorted(os.listdir(raw_dir))
for dir in dirs:
    lang_pair = dir
    src, trg = lang_pair.split("-")
    base_path = raw_dir + lang_pair + "/TED2020." + lang_pair + "."
    f_src = open(base_path+src,"r",encoding="utf8")
    f_trg = open(base_path+trg,"r",encoding="utf8")

    new_dir = target_dir+lang_pair+"/"
    os.makedirs(new_dir)
    src_train = open(new_dir + src + ".train","w",encoding="utf8")
    src_dev = open(new_dir + src + ".dev","w",encoding="utf8")
    src_test = open(new_dir + src + ".test","w",encoding="utf8")

    trg_train = open(new_dir + trg + ".train","w",encoding="utf8")
    trg_dev = open(new_dir + trg + ".dev","w",encoding="utf8")
    trg_test = open(new_dir + trg + ".test","w",encoding="utf8")

    src_lines = f_src.readlines()
    trg_lines = f_trg.readlines()
    assert len(src_lines)==len(trg_lines)
    idx = list(range(len(src_lines)))
    train_len = int(len(src_lines)*0.6)
    dev_len = int(len(src_lines)*0.2)
    random.shuffle(idx)
    for _ in range(train_len):
        i = idx[_]
        src_train.write(src_lines[i])
        trg_train.write(trg_lines[i])
    for _ in range(train_len,train_len+dev_len):
        i = idx[_]
        src_dev.write(src_lines[i])
        trg_dev.write(trg_lines[i])
    for _ in range(train_len+dev_len,len(src_lines)):
        i = idx[_]
        src_test.write(src_lines[i])
        trg_test.write(trg_lines[i])
    print(lang_pair,len(src_lines))