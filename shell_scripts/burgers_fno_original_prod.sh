export CUDA_VISIBLE_DEVICES=6

python scripts/train.py Burgers FNO_Original \
    --data_path /data/ycsong/data/zongyi/burgers_data_R10.mat \
    --n_train 1536 \
    --n_test 512 \
    --batch_size 64 \
    --raw_in_channels 1 \
    --train_subsample_rate 1 \
    --test_subsample_rate 1 \
    --n_dim 1 \
    --n_modes 21 \
    --num_prod 2 \
    --pos_encoding 0 \
    --model_pos_encoding 1 \
    --hidden_channels 64 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --scheduler_steps 100 \
    --scheduler_gamma 0.5 \
    --train_loss l2 \
    --epochs 501 \
    --verbose 1 \
    --random_seed 1 \
    --seed 0
    # --prod-layer \
    # --num_prod 1 \