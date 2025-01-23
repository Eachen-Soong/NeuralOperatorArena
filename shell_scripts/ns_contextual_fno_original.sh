export CUDA_VISIBLE_DEVICES=2

python -m scripts.train TorusVisForce FNO_Original \
    --data_path /data/ycsong/data/ns_contextual/ns_random_forces_v0.h5 \
    --n_train 100 \
    --n_test 20 \
    --raw_in_channels 3 \
    --n_dim 2 \
    --batch_size 32 \
    --train_subsample_rate 4 \
    --test_subsample_rate 4 \
    --time_step 10 \
    --n_modes 21 \
    --num_prod 0 \
    --model_pos_encoding 1 \
    --hidden_channels 32 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --scheduler_steps 100 \
    --scheduler_gamma 0.5 \
    --train_loss h1 \
    --epochs 501 \
    --verbose 1 \
    --random_seed 0 \
    --seed 0