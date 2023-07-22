import os 

dir = "./zip_data/"
trg_dir = "./raw_data/"
os.mkdir(trg_dir)
files = os.listdir(dir)
for file in files:
    if "zip" not in file:
        continue
    lang_pair = file.split(".")[0]
    file_path = dir+file
    unzip_dir = trg_dir+lang_pair
    os.mkdir(unzip_dir)
    exec_unzip = f"unzip {file_path} -d {unzip_dir}"
    os.system(exec_unzip)