python train.py \
--model_name_or_path klue/roberta-large  \
--output_dir ./models/klue/roberta-large-ep10-lr3  \
--dataset_name klue \
--dataset_config_name mrc \
--do_train  \
--train_retrieval False \
--eval_retrieval False  \
--overwrite_output_dir  \
--per_device_train_batch_size  3  \
--num_train_epochs 3 \
--learning_rate 3e-5  \
--adam_beta2 0.98

python train.py \
--model_name_or_path ./models/klue/roberta-large-ep10-lr3 \
--output_dir ./outputs/klue/roberta-large-ep10-lr3 \
--dataset_name klue \
--dataset_config_name mrc \
--do_eval  \
--train_retrieval False \
--eval_retrieval False  \
--overwrite_output_dir  \
--per_device_train_batch_size  3