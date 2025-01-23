export CUDA_VISIBLE_DEVICES=2

python scripts/train.py Burgers FNO \
    --data_path /data/ycsong/data/zongyi/burgers_data_R10.mat \
    --n_train 1536 \
    --n_test 512 \
    --batch_size 64 \
    --raw_in_channels 1 \
    --train_subsample_rate 1 \
    --test_subsample_rate 1 \
    --n_modes 21 \
    --n_layers 4 \
    --pos_encoding 0 \
    --hidden_channels 32 \
    --lifting_channels 256 \
    --projection_channels 64 \
    --rank 0.42 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --scheduler_steps 100 \
    --scheduler_gamma 0.5 \
    --train_loss h1 \
    --save_dir ./runs \
    --epochs 501 \
    --verbose 1 \
    --version_of_time 1 \
    --random_seed 0 \
    --seed 1825 \
    # --channel_mixing prod-layer \
    # --num_prod 1 \