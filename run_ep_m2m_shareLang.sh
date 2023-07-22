for seed in 2022 2023 2024
do
    python main.py --dataset europarl --mode m2m --use_adapter True --uniform False --share shareLang --local_lr 1e-3 --device cuda:0 --device_ids 0,1 --seed $seed
    python evaluation.py --dataset europarl --mode m2m --use_adapter True --uniform False --share shareLang --device cuda:0 --seed $seed
done
python compute_bleu.py --dataset europarl --mode m2m --use_adapter True --uniform False --share shareLang