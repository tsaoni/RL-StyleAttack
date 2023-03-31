# max source len should be same as the max gen target len (?)
export CUDA_VISIBLE_DEVICES=$1
if [ "$1" = 5 ]
then
    python gpt2-sentiment.py \
    --do_eval \
    --train_type forward \
    --use_cls_type victim \
    --dataset_type victim \
    --forward_model_name gpt2-medium \
    --forward_ref_model_name gpt2-medium \
    --backward_model_name gpt2-medium \
    --backward_ref_model_name gpt2-medium \
    --victim_model_name textattack/roberta-base-rotten_tomatoes \
    --attribute_model_name textattack/roberta-base-ag-news \
    --victim_dataset_name rotten_tomatoes \
    --attribute_dataset_name agnews \
    --use_cls_pipeline \
    --seed 112 \
    --learning_rate 1.41e-5 \
    --max_source_length 50 \
    --min_source_length 50 \
    --max_target_length 20 \
    --min_target_length 20 \
    --train_batch_size 32 \
    --num_train_epochs 100 \
    --max_gen_target_length 50 \
    --min_gen_target_length 20 \
    --do_sample \
    --top_k 0.0 \
    --top_p 1.0 \
    --log_with wandb \
    --load_checkpoint_path m=gpt2-medium_mt=forward_cls=textattack-roberta-base-rotten_tomatoes_clst=victim_ds=rotten_tomatoes
fi

if [ "$1" = 6 ]
then
    python gpt2-sentiment.py \
    --do_eval \
    --train_type backward \
    --use_cls_type attribute \
    --dataset_type victim \
    --forward_model_name gpt2-medium \
    --forward_ref_model_name gpt2-medium \
    --backward_model_name gpt2-medium \
    --backward_ref_model_name gpt2-medium \
    --victim_model_name textattack/roberta-base-rotten_tomatoes \
    --attribute_model_name textattack/roberta-base-ag-news \
    --victim_dataset_name rotten_tomatoes \
    --attribute_dataset_name agnews \
    --use_cls_pipeline \
    --seed 112 \
    --learning_rate 1.41e-5 \
    --max_source_length 50 \
    --min_source_length 50 \
    --max_target_length 20 \
    --min_target_length 20 \
    --train_batch_size 32 \
    --num_train_epochs 100 \
    --max_gen_target_length 50 \
    --min_gen_target_length 20 \
    --do_sample \
    --top_k 0.0 \
    --top_p 1.0 \
    --attribute_id 0 \
    --log_with wandb \
    --load_checkpoint_path m=gpt2-medium_mt=backward_cls=textattack-roberta-base-ag-news_clst=attribute_ds=rotten_tomatoes \
    # --pad_to_max_len \
fi

