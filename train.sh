python train_qlora.py \
--train_args_json chatGLM_6B_QLoRA.json \
--model_name_or_path THUDM \
--train_data_path data/train.jsonl \
--eval_data_path data/dev.jsonl \
--lora_rank 4 \
--lora_dropout 0.05 \
--compute_dtype fp32
