python3 train_NL.py --train_txt MARS_database/train_path.txt --train_info MARS_database/train_info.npy --batch_size 64 \
                     --test_txt MARS_database/test_path.txt --test_info MARS_database/test_info.npy --query_info MARS_database/query_IDX.npy \
                     --n_epochs 200 --lr 0.001 --lr_step_size 50 --optimizer adam --ckpt ckpt_NL --log_path loss.txt --class_per_batch 8 \
                     --model_type 'resnet50_NL' --num_workers 8 --track_per_class 4 --S 8 --latent_dim 2048 --temporal Done  --track_id_loss --non_layers  0 1 2 0 --stripes 16 16 16 16\
