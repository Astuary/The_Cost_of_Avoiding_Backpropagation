method=backprop

epochs=600
batch_size=8
max_batches=1000000
lr=1e-3
optimizer=adamw
momentum=0.0
checkpointing=True
dataset=agnews

# model_name=bert-base-uncased
# model_name=bert-large-uncased
# model_name=FacebookAI/roberta-base
# model_name=FacebookAI/roberta-large
model_name=meta-llama/Meta-Llama-3.1-8B
# model_name=facebook/opt-1.3b
# model_name=facebook/opt-6.7b
# model_name=facebook/opt-13b
# model_name=Qwen/Qwen2-VL-7B-Instruct

model_name_shortened=$(echo "$model_name" | cut -d'/' -f2)
max_seq_len=64
peft_method=lora
lora_r=1
lora_alpha=1

partition_count=10
dirichlet_distribution=1.0
partition_id=0

grad_norm_range=0.0
grad_value_range=0.0

accumulation_steps=1

random_seed=0
experiment_tag=flop_counter
checkpoint_interval=1

if echo "$dataset" | grep -qx "agnews"; then
    batch_size=40 #40 #8
    lr=1e-3
elif echo "$dataset" | grep -qx "boolq"; then
    batch_size=40 #4, 40
elif echo "$dataset" | grep -qx "multirc"; then
    batch_size=40 # 40, 8
elif echo "$dataset" | grep -qx "gsm8k"; then
    batch_size=6 #10/8 6 4
    lr=1e-5
    optimizer=adamw
elif echo "$dataset" | grep -qx "mmlu"; then
    batch_size=6 #8, 6
    lr=1e-4
    optimizer=sgd_nesterov
    momentum=0.9
elif echo "$dataset" | grep -qx "vqav2"; then
    batch_size=6 #8 6
    optimizer=sgd
elif echo "$dataset" | grep -qx "textvqa"; then
    batch_size=8
fi

experiment_name=${epochs}_e_${batch_size}_bs_${lr}_lr_${peft_method}_${lora_r}_r_${lora_alpha}_alp_${accumulation_steps}_accumulation_steps_${checkpointing}_checkpointing_${grad_norm_range//./_}_gradnormrange_${grad_value_range//./_}_gradvalrange_${optimizer}_opt_${momentum//.}_m_${random_seed}_seed_${experiment_tag}
results_dir=./results/${dataset}/${method}/${model_name_shortened}/${experiment_name}/
echo "Starting $method on $dataset, Experiment $experiment_name"

mkdir -p $results_dir 
cp backprop.py $results_dir

python3 backprop.py \
    --partition_count $partition_count --epochs $epochs \
    --batch_size $batch_size --lr $lr --optimizer $optimizer \
    --max_batches $max_batches --momentum $momentum \
    --model_name $model_name --max_seq_len $max_seq_len \
    --peft_method $peft_method --lora_r $lora_r --lora_alpha $lora_alpha \
    --dataset $dataset --dirichlet_distribution $dirichlet_distribution \
    --partition_id $partition_id --random_seed $random_seed \
    --checkpoint_interval $checkpoint_interval --experiment_name $experiment_name \
    --results_dir $results_dir --accumulation_steps $accumulation_steps \
    --checkpointing $checkpointing
