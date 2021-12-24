python train.py \
--model_name_or_path klue/roberta-base  \
--output_dir ./models/klue/roberta-base  \
--dataset_name klue \
--dataset_config_name mrc \
--do_train  \
--train_retrieval False \
--eval_retrieval False  \
--overwrite_output_dir  \
#--per_device_train_batch_size  3

python train.py \
--model_name_or_path ./models/klue/roberta-base \
--output_dir ./outputs/klue/roberta-base  \
--dataset_name klue \
--dataset_config_name mrc \
--do_eval  \
--train_retrieval False \
--eval_retrieval False  \
--overwrite_output_dir  \
#--per_device_train_batch_size  4