export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./log" ]; then
    mkdir ./log
fi

if [ ! -d "./log/CMamba" ]; then
    mkdir ./log/CMamba
fi

if [ ! -d "./log/CMamba/weather" ]; then
    mkdir ./log/CMamba/weather
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
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --e_layers 3 \
  --lradj type3 \
  --train_epochs 100 \
  --patience 3 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --batch_size 64 \
  --pscan \
  --dropout 0.0 \
  --head_dropout 0.0 \
  --learning_rate 0.001 \
  --d_model 128 \
  --d_ff 128 \
  --num_workers 1 \
  --channel_mixup \
  --sigma 0.1 \
  --gddmlp \
  --d_state 16 \
  --avg \
  --max \
  --reduction 2 \
  --seed $seed \
  --itr 1 | tee -a ./log/CMamba/weather/$seq_len'_'$pred_len.txt
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
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --e_layers 3 \
  --lradj type3 \
  --train_epochs 100 \
  --patience 3 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --batch_size 64 \
  --pscan \
  --dropout 0.0 \
  --head_dropout 0.0 \
  --learning_rate 0.0001 \
  --d_model 128 \
  --d_ff 128 \
  --num_workers 1 \
  --channel_mixup \
  --sigma 0.1 \
  --gddmlp \
  --d_state 16 \
  --avg \
  --max \
  --reduction 2 \
  --seed $seed \
  --itr 1 | tee -a ./log/CMamba/weather/$seq_len'_'$pred_len.txt
done
done
done

for seq_len in 96
do
for pred_len in 336
do
for seed in 2021
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --e_layers 3 \
  --lradj type3 \
  --train_epochs 100 \
  --patience 3 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --batch_size 64 \
  --pscan \
  --dropout 0.0 \
  --head_dropout 0.0 \
  --learning_rate 0.001 \
  --d_model 128 \
  --d_ff 128 \
  --num_workers 1 \
  --channel_mixup \
  --sigma 0.1 \
  --gddmlp \
  --d_state 16 \
  --avg \
  --max \
  --reduction 2 \
  --seed $seed \
  --itr 1 | tee -a ./log/CMamba/weather/$seq_len'_'$pred_len.txt
done
done
done

for seq_len in 96
do
for pred_len in 720
do
for seed in 2021
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --e_layers 3 \
  --lradj type3 \
  --train_epochs 100 \
  --patience 3 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --batch_size 64 \
  --pscan \
  --dropout 0.0 \
  --head_dropout 0.0 \
  --learning_rate 0.0001 \
  --d_model 128 \
  --d_ff 128 \
  --num_workers 1 \
  --channel_mixup \
  --sigma 0.1 \
  --gddmlp \
  --d_state 16 \
  --avg \
  --max \
  --reduction 2 \
  --seed $seed \
  --itr 1 | tee -a ./log/CMamba/weather/$seq_len'_'$pred_len.txt
done
done
done
