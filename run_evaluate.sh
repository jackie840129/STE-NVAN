TEST_TXT=./MARS_database/test_path.txt
TEST_INFO=./MARS_database/test_info.npy
QUERY_INFO=./MARS_database/query_IDX.npy

# Evaluate ResNet50 + FPL (mean or max)
#LOAD_CKPT=./ckpt/R50_baseline_mean.pth
#python3 evaluate.py --test_txt $TEST_TXT --test_info $TEST_INFO --query_info $QUERY_INFO \
                          #--batch_size 64 --model_type 'resnet50_s1' --num_workers 8  --S 8 \
                          #--latent_dim 2048 --temporal mean --stride 1 --load_ckpt $LOAD_CKPT 
#Evaluate NVAN (R50 + 5 NL + FPL)
LOAD_CKPT=./ckpt/NVAN.pth
python3 evaluate.py --test_txt $TEST_TXT  --test_info  $TEST_INFO   --query_info $QUERY_INFO \
                    --batch_size 64 --model_type 'resnet50_NL' --num_workers 8  --S 8 --latent_dim 2048 \
                    --temporal Done  --non_layers  0 2 3 0 --load_ckpt $LOAD_CKPT \

# Evaluate NVAN (R50 + 5 NL + Stripe + Hierarchical + FPL)
#LOAD_CKPT=./ckpt/STE_NVAN.pth
#python3 evaluate.py --test_txt $TEST_TXT  --test_info  $TEST_INFO   --query_info $QUERY_INFO \
                    #--batch_size 128 --model_type 'resnet50_NL_stripe_hr' --num_workers 8  --S 8 --latent_dim 2048 \
                    #--temporal Done  --non_layers  0 2 3 0 --stripe 16 16 16 16 --load_ckpt $LOAD_CKPT \
