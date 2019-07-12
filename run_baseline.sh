python3 train_baseline.py --train_txt MARS_database/train_path.txt --train_info MARS_database/train_info.npy --batch_size 128\
                     --test_txt MARS_database/test_path.txt --test_info MARS_database/test_info.npy --query_info MARS_database/query_IDX.npy \
                     --n_epochs 300 --lr 0.0001 --lr_step_size 50 --optimizer adam --ckpt ckpt_baseline_test --log_path loss.txt --class_per_batch 2 \
                     --model_type 'resnet50_s1' --num_workers 8 --track_per_class 4 --S 8 --latent_dim 2048 --temporal max  --track_id_loss --stride 1 \
