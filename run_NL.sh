TRAIN_TXT=./MARS_database/train_path.txt
TRAIN_INFO=./MARS_database/train_info.npy
TEST_TXT=./MARS_database/test_path.txt
TEST_INFO=./MARS_database/test_info.npy
QUERY_INFO=./MARS_database/query_IDX.npy

# For NVAN
CKPT=ckpt_NL_0230
python3 train_NL.py --train_txt $TRAIN_TXT --train_info $TRAIN_INFO  --batch_size 64 \
                     --test_txt $TEST_TXT  --test_info  $TEST_INFO   --query_info $QUERY_INFO \
                     --n_epochs 200 --lr 0.0001 --lr_step_size 50 --optimizer adam --ckpt $CKPT --log_path loss.txt --class_per_batch 8 \
                     --model_type 'resnet50_NL' --num_workers 8 --track_per_class 4 --S 8 --latent_dim 2048 --temporal Done  --track_id_loss \
                     --non_layers  0 2 3 0  

# For STE-NVAN
#CKPT=ckpt_NL_stripe16_hr_0230
#python3 train_NL.py --train_txt $TRAIN_TXT --train_info $TRAIN_INFO  --batch_size 64 \
                     #--test_txt $TEST_TXT  --test_info  $TEST_INFO   --query_info $QUERY_INFO \
                     #--n_epochs 200 --lr 0.0001 --lr_step_size 50 --optimizer adam --ckpt $CKPT --log_path loss.txt --class_per_batch 8 \
                     #--model_type 'resnet50_NL_stripe_hr' --num_workers 8 --track_per_class 4 --S 8 --latent_dim 2048 --temporal Done  --track_id_loss \
                     #--non_layers  0 2 3 0 --stripes 16 16 16 16 
