method=zero_order_finite_differences

epochs=600
batch_size=8
max_batches=100000
lr=1e-3
optimizer=adamw
momentum=0.0
dataset=agnews
calibrated=False

# model_name=bert-base-uncased
# model_name=bert-large-uncased
# model_name=FacebookAI/roberta-base
# model_name=roberta-large
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

perturbation_count=1
perturbation_var=0.01
perturbation_mean=0
fixed_seed=False
jvp_range=0.0
grad_norm_range=0.0
grad_value_range=0.0

prob_to_keep=1.0
accumulation_steps=1

random_seed=0
experiment_tag=flop_counter
checkpoint_interval=1

if echo "$dataset" | grep -qx "agnews"; then
    batch_size=40 #40 #8
    max_batches=100000 #00
    lr=1e-4
elif echo "$dataset" | grep -qx "boolq"; then
    batch_size=40 #4, 40
    max_batches=10000
    lr=1e-4
    optimizer=adamw
elif echo "$dataset" | grep -qx "multirc"; then
    batch_size=40 # 40, 8
    max_batches=10000
    lr=1e-3
    optimizer=adamw #sgd
elif echo "$dataset" | grep -qx "gsm8k"; then
    batch_size=4 #10/8 6 4
    max_batches=1000000
    lr=1e-5 #4
    optimizer=sgd_nesterov #adamw
    momentum=0.9
elif echo "$dataset" | grep -qx "mmlu"; then
    batch_size=10 #12, 6
    max_batches=1000
    lr=1e-5
    optimizer=adamw #sgd_nesterov
    momentum=0.0
elif echo "$dataset" | grep -qx "vqav2"; then
    batch_size=6 #8 6
    max_batches=1000
    lr=1e-3
    optimizer=adamw #sgd
elif echo "$dataset" | grep -qx "gqa"; then
    batch_size=4 # 6
    max_batches=1000
elif echo "$dataset" | grep -qx "textvqa"; then
    batch_size=8
    max_batches=1000
    lr=1e-4
    optimizer=sgd
fi

if echo "$method" | grep -qx "svrg_zero_order_finite_differences"; then
    experiment_name=${epochs}_e_${batch_size}_bs_${lr}_lr_${peft_method}_${lora_r}_r_${lora_alpha}_alp_${fixed_seed}_seed_${accumulation_steps}_accumulation_steps_${perturbation_count}_pcount_${perturbation_var//./_}_pvar_${perturbation_mean//./_}_pmean_${jvp_range//./_}_jvprange_${grad_norm_range//./_}_gradnormrange_${grad_value_range//./_}_gradvalrange_${optimizer}_opt_${momentum//.}_m_${random_seed}_seed_${experiment_tag}
    results_dir=./results/${dataset}/${method}/${model_name_shortened}/${experiment_name}/
    echo "Starting $method on $dataset, Experiment $experiment_name"
    
    mkdir -p $results_dir 
    cp ./trainers/svrg_zero_order_finite_differences.py $results_dir
    
    python3 ./trainers/svrg_zero_order_finite_differences.py \
        --partition_count $partition_count --epochs $epochs \
        --batch_size $batch_size --lr $lr --optimizer $optimizer \
        --max_batches $max_batches --momentum $momentum\
        --model_name $model_name --max_seq_len $max_seq_len \
        --peft_method $peft_method --lora_r $lora_r --lora_alpha $lora_alpha \
        --dataset $dataset --dirichlet_distribution $dirichlet_distribution \
        --partition_id $partition_id --perturbation_count $perturbation_count \
        --perturbation_step_size $perturbation_var --perturbation_mean $perturbation_mean \
        --fixed_seed $fixed_seed --random_seed $random_seed \
        --checkpoint_interval $checkpoint_interval --experiment_name $experiment_name \
        --results_dir $results_dir --jvp_range $jvp_range \
        --grad_norm_range $grad_norm_range --grad_value_range $grad_value_range \
        --prob_to_keep $prob_to_keep --accumulation_steps $accumulation_steps
elif echo "$method" | grep -qx "sparse_zero_order_finite_differences"; then
    experiment_name=${epochs}_e_${batch_size}_bs_${lr}_lr_${peft_method}_${lora_r}_r_${lora_alpha}_alp_${fixed_seed}_seed_${accumulation_steps}_accumulation_steps_${perturbation_count}_pcount_${perturbation_var//./_}_pvar_${perturbation_mean//./_}_pmean_${jvp_range//./_}_jvprange_${grad_norm_range//./_}_gradnormrange_${grad_value_range//./_}_gradvalrange_${optimizer}_opt_${momentum//.}_m_${random_seed}_${calibrated}_adaptive_seed_${experiment_tag}
    results_dir=./results/${dataset}/${method}/${model_name_shortened}/${experiment_name}/
    echo "Starting $method on $dataset, Experiment $experiment_name"
    
    mkdir -p $results_dir 
    cp ./trainers/sparse_zero_order_finite_differences.py $results_dir
    
    python3 ./trainers/sparse_zero_order_finite_differences.py \
        --partition_count $partition_count --epochs $epochs \
        --batch_size $batch_size --lr $lr --optimizer $optimizer \
        --max_batches $max_batches --momentum $momentum\
        --model_name $model_name --max_seq_len $max_seq_len \
        --peft_method $peft_method --lora_r $lora_r --lora_alpha $lora_alpha \
        --dataset $dataset --dirichlet_distribution $dirichlet_distribution \
        --partition_id $partition_id --perturbation_count $perturbation_count \
        --perturbation_step_size $perturbation_var --perturbation_mean $perturbation_mean \
        --fixed_seed $fixed_seed --random_seed $random_seed \
        --checkpoint_interval $checkpoint_interval --experiment_name $experiment_name \
        --results_dir $results_dir --jvp_range $jvp_range \
        --grad_norm_range $grad_norm_range --grad_value_range $grad_value_range \
        --prob_to_keep $prob_to_keep --accumulation_steps $accumulation_steps 
elif echo "$method" | grep -qx "zero_order_finite_differences"; then
    experiment_name=${epochs}_e_${batch_size}_bs_${lr}_lr_${peft_method}_${lora_r}_r_${lora_alpha}_alp_${fixed_seed}_seed_${accumulation_steps}_accumulation_steps_${perturbation_count}_pcount_${perturbation_var//./_}_pvar_${perturbation_mean//./_}_pmean_${jvp_range//./_}_jvprange_${grad_norm_range//./_}_gradnormrange_${grad_value_range//./_}_gradvalrange_${optimizer}_opt_${momentum//.}_m_${random_seed}_${calibrated}_adaptive_seed_${experiment_tag}
    results_dir=./results/${dataset}/${method}/${model_name_shortened}/${experiment_name}/
    echo "Starting $method on $dataset, Experiment $experiment_name"
    
    mkdir -p $results_dir 
    cp ./trainers/zero_order_finite_differences.py $results_dir
    
    python3 ./trainers/zero_order_finite_differences.py \
        --partition_count $partition_count --epochs $epochs \
        --batch_size $batch_size --lr $lr --optimizer $optimizer \
        --max_batches $max_batches --momentum $momentum\
        --model_name $model_name --max_seq_len $max_seq_len \
        --peft_method $peft_method --lora_r $lora_r --lora_alpha $lora_alpha \
        --dataset $dataset --dirichlet_distribution $dirichlet_distribution \
        --partition_id $partition_id --perturbation_count $perturbation_count \
        --perturbation_step_size $perturbation_var --perturbation_mean $perturbation_mean \
        --fixed_seed $fixed_seed --random_seed $random_seed \
        --checkpoint_interval $checkpoint_interval --experiment_name $experiment_name \
        --results_dir $results_dir --jvp_range $jvp_range \
        --grad_norm_range $grad_norm_range --grad_value_range $grad_value_range \
        --prob_to_keep $prob_to_keep --accumulation_steps $accumulation_steps \
        --calibrated $calibrated
elif echo "$method" | grep -qx "mezo"; then
    experiment_name=${epochs}_e_${batch_size}_bs_${lr}_lr_${peft_method}_${lora_r}_r_${lora_alpha}_alp_${fixed_seed}_seed_${accumulation_steps}_accumulation_steps_${perturbation_count}_pcount_${perturbation_var//./_}_pvar_${perturbation_mean//./_}_pmean_${jvp_range//./_}_jvprange_${grad_norm_range//./_}_gradnormrange_${grad_value_range//./_}_gradvalrange_${optimizer}_opt_${momentum//.}_m_${random_seed}_seed_${experiment_tag}
    results_dir=./results/${dataset}/${method}/${model_name_shortened}/${experiment_name}/
    echo "Starting $method on $dataset, Experiment $experiment_name"
    
    mkdir -p $results_dir 
    cp ./trainers/mezo.py $results_dir
    
    python3 ./trainers/mezo.py \
        --partition_count $partition_count --epochs $epochs \
        --batch_size $batch_size --lr $lr --optimizer $optimizer \
        --max_batches $max_batches --momentum $momentum\
        --model_name $model_name --max_seq_len $max_seq_len \
        --peft_method $peft_method --lora_r $lora_r --lora_alpha $lora_alpha \
        --dataset $dataset --dirichlet_distribution $dirichlet_distribution \
        --partition_id $partition_id --perturbation_count $perturbation_count \
        --perturbation_step_size $perturbation_var --perturbation_mean $perturbation_mean \
        --fixed_seed $fixed_seed --random_seed $random_seed \
        --checkpoint_interval $checkpoint_interval --experiment_name $experiment_name \
        --results_dir $results_dir --jvp_range $jvp_range \
        --grad_norm_range $grad_norm_range --grad_value_range $grad_value_range \
        --prob_to_keep $prob_to_keep --accumulation_steps $accumulation_steps
fi
