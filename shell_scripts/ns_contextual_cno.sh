export CUDA_VISIBLE_DEVICES=1

python -m scripts.train TorusVisForce CNO \
    --data_path /data/ycsong/data/ns_contextual/ns_random_forces_v0.h5 \
    --n_train 100 \
    --n_test 20 \
    --raw_in_channels 3 \
    --out_channels 1 \
    --n_dim 2 \
    --batch_size 64 \
    --train_subsample_rate 4 \
    --test_subsample_rate 4 \
    --time_step 4 \
    --spatial_size 64 \
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

