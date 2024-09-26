export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./log" ]; then
    mkdir ./log
fi

if [ ! -d "./log/CMamba" ]; then
    mkdir ./log/CMamba
fi

if [ ! -d "./log/CMamba/ettm2" ]; then
    mkdir ./log/CMamba/ettm2
fi

model_name=CMamba


for seed in 2021
do
for seq_len in 96
do
for pred_len in 96
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTm2 \
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
  --sigma 1.0 \
  --gddmlp \
  --avg \
  --max \
  --reduction 2 \
  --seed $seed \
  --itr 1 | tee -a ./log/CMamba/ettm2/$seq_len'_'$pred_len.txt
done
done
done


for seed in 2021
do
for seq_len in 96
do
for pred_len in 192
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTm2 \
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
  --sigma 1.0 \
  --gddmlp \
  --avg \
  --max \
  --reduction 2 \
  --seed $seed \
  --itr 1 | tee -a ./log/CMamba/ettm2/$seq_len'_'$pred_len.txt
done
done
done


for seed in 2021
do
for seq_len in 96
do
for pred_len in 336
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --lradj type3 \
  --patience 3 \
  --train_epochs 100 \
  --e_layers 3 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 128 \
  --d_ff 128 \
  --des 'Exp' \
  --pscan \
  --dropout 0.1 \
  --head_dropout 0.1 \
  --learning_rate 0.001 \
  --batch_size 64 \
  --num_workers 1 \
  --channel_mixup \
  --sigma 1.0 \
  --gddmlp \
  --avg \
  --max \
  --reduction 2 \
  --seed $seed \
  --itr 1 | tee -a ./log/CMamba/ettm2/$seq_len'_'$pred_len.txt
done
done
done


for seed in 2021
do
for seq_len in 96
do
for pred_len in 720
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --lradj type3 \
  --patience 3 \
  --train_epochs 100 \
  --e_layers 3 \
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
  --learning_rate 0.0001 \
  --batch_size 64 \
  --num_workers 1 \
  --channel_mixup \
  --sigma 1.0 \
  --gddmlp \
  --avg \
  --max \
  --reduction 2 \
  --seed $seed \
  --itr 1 | tee -a ./log/CMamba/ettm2/$seq_len'_'$pred_len.txt
done
done
done
