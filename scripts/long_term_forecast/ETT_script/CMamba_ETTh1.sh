export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./log" ]; then
    mkdir ./log
fi

if [ ! -d "./log/CMamba" ]; then
    mkdir ./log/CMamba
fi

if [ ! -d "./log/CMamba/etth1" ]; then
    mkdir ./log/CMamba/etth1
fi

model_name=CMamba

for seq_len in 96
do
for pred_len in 96 336 720
do
for seed in 2021
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTh1 \
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
  --learning_rate 0.0005 \
  --batch_size 64 \
  --num_workers 1 \
  --channel_mixup \
  --sigma 1.0 \
  --gddmlp \
  --avg \
  --max \
  --reduction 2 \
  --seed $seed \
  --itr 1 | tee -a ./log/CMamba/etth1/$seq_len'_'$pred_len.txt
done
done
done

for seq_len in 96
do
for pred_len in 192
do
for seed in 2021
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --lradj type3 \
  --patience 3 \
  --train_epochs 100 \
  --e_layers 2 \
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
  --learning_rate 0.0005 \
  --batch_size 64 \
  --num_workers 1 \
  --channel_mixup \
  --sigma 1.0 \
  --gddmlp \
  --avg \
  --max \
  --reduction 2 \
  --seed $seed \
  --itr 1 | tee -a ./log/CMamba/etth1/$seq_len'_'$pred_len.txt
done
done
done
