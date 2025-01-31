export CUDA_VISIBLE_DEVICES=3

python -m scripts.predict_multitask MultiTaskCylinderFlow FNO \
    --train_path /data/jmwang/DimOL/2D_cylinders/train_data_new/circle_1 \
                 /data/jmwang/DimOL/2D_cylinders/train_data_new/triangle_1 \
                 /data/jmwang/DimOL/2D_cylinders/train_data_new/square_1 \
                 /data/jmwang/DimOL/2D_cylinders/train_data_new/square_1_triangle_1 \
    --test_path  /data/jmwang/DimOL/2D_cylinders/test_data_new/circle_1 \
                 /data/jmwang/DimOL/2D_cylinders/test_data_new/triangle_1 \
                 /data/jmwang/DimOL/2D_cylinders/test_data_new/square_1 \
                 /data/jmwang/DimOL/2D_cylinders/test_data_new/square_1_triangle_1 \
                 /data/jmwang/DimOL/2D_cylinders/test_data_new/circle_1_triangle_1 \
                 /data/jmwang/DimOL/2D_cylinders/test_data_new/circle_1_square_1 \
    --n_train 1 \
    --n_test 1 \
    --raw_in_channels 3 \
    --out_channels 2 \
    --pos_encoding 0 \
    --n_dim 2 \
    --batch_size 32 \
    --train_subsample_rate 1 \
    --test_subsample_rate 1 \
    --time_step 10 \
    --load_path ./runs/MultiTaskCylinderFlow/FNO/exp_1-27-21-41/version_0 \
    --log_input 1
 