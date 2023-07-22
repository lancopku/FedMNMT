# FedMNMT
[Findings of ACL 2023] Communication Efficient Federated Learning for Multilingual Machine Translation with Adapter
The code is coming soon. [[pdf]](https://arxiv.org/pdf/2305.12449.pdf)

# Data prepare
To prepare Europarl dataset, run the following instructions.
```
cd data/Europarl/zip_data
bash download_data.sh
cd ..
python unzip_data.py
python data_preprocess.py
```
The preparation for TED2020 dataset is similar.

# Run experiments
To train and evaluate adapter-families on Europarl dataset, run
```
bash run_ep_m2m_shareLang.sh
```
To run experiments on TED2020 dataset, change the arguments --dataset and --mode to "ted2020" and "m2en" respectively.

To train adapter-fed/random/gradients, change the argument --share to "shareAll"/"random"/"shareFix".