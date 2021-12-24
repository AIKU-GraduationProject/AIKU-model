python train.py \
--model_name_or_path ./models/koelectra-base-v3-discriminator \
--output_dir ./outputs/koelectra-base-v3-discriminator  \
--dataset_name klue \
--dataset_config_name mrc \
--do_eval  \
--train_retrieval False \
--eval_retrieval False  \
#--per_device_train_batch_size  4