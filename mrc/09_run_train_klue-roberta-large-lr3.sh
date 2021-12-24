#python train.py \
#--model_name_or_path /home/konkuk/Desktop/koo/MRC/models  \
#--output_dir ./models/klue/roberta-large-ab8  \
#--dataset_name klue \
#--dataset_config_name mrc \
#--do_train  \
#--train_retrieval False \
#--eval_retrieval False  \
#--overwrite_output_dir  \
#--per_device_train_batch_size  3  \
#--adam_beta1 0.8  \
#--learning_rate 3e-5

python train.py \
--model_name_or_path /home/konkuk/Desktop/koo/MRC/models \
--output_dir ./outputs/klue/roberta-large-ab8 \
--dataset_name klue \
--dataset_config_name mrc \
--do_eval  \
--train_retrieval False \
--eval_retrieval False  \
--overwrite_output_dir  \
--per_device_train_batch_size  3