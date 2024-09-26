export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./log" ]; then
    mkdir ./log
fi

if [ ! -d "./log/CMamba" ]; then
    mkdir ./log/CMamba
fi

if [ ! -d "./log/CMamba/ettm1" ]; then
    mkdir ./log/CMamba/ettm1
fi

model_name=CMamba

for seq_len in 96
do
for pred_len in 96 192
do
for seed in 2021
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --lradj type3 \
  --patience 3 \
  --train_epochs 100 \
  --e_layers 4 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 128 \
  --d_ff 128 \
  --des 'Exp' \
  --pscan \
  --dropout 0.0 \
  --head_dropout 0.0 \
  --learning_rate 0.0005 \
  --batch_size 64 \
  --num_workers 1 \
  --channel_mixup \
  --sigma 1 \
  --gddmlp \
  --avg \
  --max \
  --reduction 2 \
  --seed $seed \
  --itr 1 | tee -a ./log/CMamba/ettm1/$seq_len'_'$pred_len.txt
done
done
done


for seq_len in 96
do
for pred_len in 336 720
do
for seed in 2021
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --lradj type3 \
  --patience 3 \
  --train_epochs 100 \
  --e_layers 4 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 128 \
  --d_ff 128 \
  --des 'Exp' \
  --pscan \
  --dropout 0.0 \
  --head_dropout 0.0 \
  --learning_rate 0.001 \
  --batch_size 64 \
  --num_workers 1 \
  --channel_mixup \
  --sigma 1 \
  --gddmlp \
  --avg \
  --max \
  --reduction 2 \
  --seed $seed \
  --itr 1 | tee -a ./log/CMamba/ettm1/$seq_len'_'$pred_len.txt
done
done
done
