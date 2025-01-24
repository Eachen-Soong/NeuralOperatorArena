export CUDA_VISIBLE_DEVICES=4

python -m scripts.train_multitask MultiTaskCylinderFlow FNO \
    --data_path /data/jmwang/DimOL/2D_cylinders/ \
    --n_train 64 \
    --n_test 16 \
    --raw_in_channels 3 \
    --out_channels 2 \
    --n_dim 2 \
    --batch_size 32 \
    --train_subsample_rate 1 \
    --test_subsample_rate 1 \
    --time_step 10 \
    --n_modes 21 \
    --channel_mixing mlp \
    --mixing_layers 4 \
    --num_prod 0 \
    --n_layers 4 \
    --pos_encoding 0 \
    --hidden_channels 32 \
    --lifting_channels 256 \
    --projection_channels 64 \
    --factorization tucker \
    --rank 0.42 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --scheduler_steps 100 \
    --scheduler_gamma 0.5 \
    --train_loss h1 \
    --epochs 1 \
    --verbose 1 \
    --random_seed 0 \
    --seed 0

