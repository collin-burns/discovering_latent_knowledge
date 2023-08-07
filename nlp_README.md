# Abstract
Additional Info relevant to the datasets we use in this project
# DataSets:
>1. jigsaw dataset - https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
>2. jigsaw_unintended_bias dataset - https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data

## Prompts:

>1. jigsaw_unintended_bias indices 
 >  2. 1 - moderator
  > 4. 5 - browsing
 >  5. 6 - binary_pred
 >  6. 8 - yes_or_no
  > 7. 9 - yes_or_no_remove

# Generate
>Running commands:
>1. jigsaw: 
>   2. python generate.py --model_name roberta-mnli --num_examples 100 --batch_size 20 --dataset_name jigsaw_toxicity_pred
>2. jigsaw_unintended_bias:  
>   3. python generate.py --model_name roberta-mnli --num_examples 100 --batch_size 20 --dataset_name jigsaw_unintended_bias --device cpu --dataset_dir .\ --split train --prompt_idx 8
