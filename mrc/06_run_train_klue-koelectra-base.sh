python train.py \
--model_name_or_path seongju/klue-mrc-koelectra-base  \
--output_dir ./outputs/klue-mrc-koelectra-base \
--dataset_name klue \
--dataset_config_name mrc \
--do_eval  \
--train_retrieval False \
--eval_retrieval False
