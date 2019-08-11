TRAIN_TXT=./MARS_database/train_path.txt
TRAIN_INFO=./MARS_database/train_info.npy
TEST_TXT=./MARS_database/test_path.txt
TEST_INFO=./MARS_database/test_info.npy
QUERY_INFO=./MARS_database/query_IDX.npy

CKPT=ckpt_baseline_mean
python3 train_baseline.py --train_txt $TRAIN_TXT --train_info $TRAIN_INFO  --batch_size 64 \
                          --test_txt $TEST_TXT  --test_info $TEST_INFO --query_info $QUERY_INFO \
                          --n_epochs 300 --lr 0.0001 --lr_step_size 50 --optimizer adam --ckpt $CKPT --log_path loss.txt \
                          --model_type 'resnet50_s1' --num_workers 8 --class_per_batch 8 --track_per_class 4 --S 8 \
                          --latent_dim 2048 --temporal mean  --track_id_loss --stride 1 \
