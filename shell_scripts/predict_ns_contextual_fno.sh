export CUDA_VISIBLE_DEVICES=4

python -m scripts.predict TorusVisForce FNO \
    --data_path /data/ycsong/data/ns_contextual/ns_random_forces_top20_mu.h5 \
    --n_train 0 \
    --n_test 1 \
    --batch_size 1024 \
    --test_subsample_rate 4 \
    --time_step 4 \
    --load_path ./runs/TorusVisForce/FNO/exp_1-23-11-16/version_0 \
