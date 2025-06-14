export CUDA_VISIBLE_DEVICES=6

python -m scripts.train TorusLi FNO_Original \
    --data_path /data/ycsong/data/zongyi/NavierStokes_V1e-5_N1200_T20.mat \
    --n_train 100 \
    --n_test 20 \
    --raw_in_channels 1 \
    --n_dim 2 \
    --batch_size 32 \
    --train_subsample_rate 1 \
    --test_subsample_rate 1 \
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
