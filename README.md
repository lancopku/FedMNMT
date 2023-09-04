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

# Update
In section 5.3, keeping output-end adapters reaches the highest scores among three pruning strategies. 

Actually, we also conduct this analysis experiment with method adapter-fed and keeping input-end adapters is the best strategy in this setting (results are shown in the following table). We feel sorry that it is not rigorous to directly say "adapters in the top layers play more important roles" in our paper. 

| adapters     | Macro Avg. | Micro Avg. |
| --------     | :--------: | :--------: |
| input-end    | 23.68      | 23.54      |
| middle-layer | 23.37      | 23.37      |
| output-end   | 22.84      | 23.00      |

However, considering method adapter-fed is a very weak baseline with poor performance, the analysis experiment on this method is relatively meaningless and less significant. If we want to reach as competitive performance as possible while further reducing communication costs, keeping output-end adapters in method adapter-families is actually a better strategy. From this aspect, it is not wrong to conclude that keeping output-end adapters is the best among these three strategies. For the reasons above, we only show the results of experiment with method adapter-families in section 5.3 and draw a conclusion that adapters in the top layers are more important.