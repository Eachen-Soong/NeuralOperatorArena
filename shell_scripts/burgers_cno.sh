export CUDA_VISIBLE_DEVICES=4

python scripts/train.py Burgers CNO \
    --data_path  /data/ycsong/data/zongyi/burgers_data_R10.mat \
    --n_train 1536 \
    --n_test 512 \
    --batch_size 64 \
    --n_dim 1 \
    --raw_in_channels 1 \
    --train_subsample_rate 1 \
    --test_subsample_rate 1 \
    --spatial_size 8192 \
    --n_layers 4 \
    --n_res 4 \
    --n_res_neck 16 \
    --channel_multiplier 16 \
    --use_bn 1 \
    --pos_encoding 0 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --scheduler_steps 100 \
    --scheduler_gamma 0.5 \
    --train_loss l2 \
    --epochs 501 \
    --verbose 1 \
    --random_seed 0 \
    --seed 0
