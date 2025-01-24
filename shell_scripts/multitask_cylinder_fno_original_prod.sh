export CUDA_VISIBLE_DEVICES=7

python -m scripts.train_multitask MultiTaskCylinderFlow FNO_Original \
    --train_path /data/jmwang/DimOL/2D_cylinders/train_data_new/circle_1 \
                 /data/jmwang/DimOL/2D_cylinders/train_data_new/triangle_1 \
    --test_path  /data/jmwang/DimOL/2D_cylinders/test_data_new/circle_1 \
                 /data/jmwang/DimOL/2D_cylinders/test_data_new/triangle_1 \
                 /data/jmwang/DimOL/2D_cylinders/test_data_new/circle_1_triangle_1 \
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
    --num_prod 2 \
    --model_pos_encoding 1 \
    --hidden_channels 32 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --scheduler_steps 100 \
    --scheduler_gamma 0.5 \
    --train_loss h1 \
    --epochs 501 \
    --verbose 1 \
    --random_seed 1 \
    --seed 0

