export CUDA_VISIBLE_DEVICES=7

python -m scripts.train_dim TorusVisForceDim FNO \
    --data_path /data/ycsong/data/ns_contextual/ns_random_forces_v0.h5 \
    --n_train 1000 \
    --n_test 200 \
    --raw_in_channels 3 \
    --raw_in_consts 1 \
    --n_dim 2 \
    --batch_size 32 \
    --train_subsample_rate 4 \
    --test_subsample_rate 4 \
    --predict_feature u \
    --time_step 1 \
    --time_skips 10 \
    --n_modes 21 \
    --channel_mixing mlp \
    --mixing_layers 4 \
    --n_layers 4 \
    --pos_encoding 1 \
    --hidden_channels 32 \
    --lifting_channels 256 \
    --projection_channels 64 \
    --factorization tucker \
    --rank 0.42 \
    --norm dim_norm \
    --preactivation 1 \
    --prediction_dims 0 \
    --num_consts 3 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --scheduler_steps 100 \
    --scheduler_gamma 0.5 \
    --train_loss h1 \
    --epochs 501 \
    --verbose 1 \
    --random_seed 0 \
    --seed 0

