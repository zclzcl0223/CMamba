export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./log" ]; then
    mkdir ./log
fi

model_name=CMamba

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
  --label_len 48 \
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
  --dropout 0.1 \
  --head_dropout 0.1 \
  --learning_rate 0.001 \
  --d_model 128 \
  --d_ff 128 \
  --num_workers 1 \
  --channel_mixup \
  --sigma 0.1 \
  --channel_att \
  --avg \
  --max \
  --reduction 2 \
  --itr 1 >> ./log/log_weather_$seq_len'_'$pred_len.txt
done
done


for seq_len in 96
do
for pred_len in 192
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
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 3 \
  --lradj type3 \
  --train_epochs 100 \
  --patience 10 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --batch_size 64 \
  --pscan \
  --dropout 0.1 \
  --head_dropout 0.1 \
  --learning_rate 0.0001 \
  --d_model 128 \
  --d_ff 128 \
  --num_workers 1 \
  --channel_mixup \
  --sigma 0.1 \
  --channel_att \
  --avg \
  --max \
  --reduction 2 \
  --itr 1 >> ./log/log_weather_$seq_len'_'$pred_len.txt
done
done


for seq_len in 96
do
for pred_len in 336
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
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 3 \
  --lradj type3 \
  --train_epochs 100 \
  --patience 10 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --batch_size 64 \
  --pscan \
  --dropout 0.1 \
  --head_dropout 0.1 \
  --learning_rate 0.0001 \
  --d_model 128 \
  --d_ff 128 \
  --num_workers 1 \
  --channel_mixup \
  --sigma 0.5 \
  --channel_att \
  --avg \
  --max \
  --reduction 2 \
  --itr 1 >> ./log/log_weather_$seq_len'_'$pred_len.txt
done
done


for seq_len in 96
do
for pred_len in 720
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
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 3 \
  --lradj type3 \
  --train_epochs 100 \
  --patience 10 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --batch_size 64 \
  --pscan \
  --dropout 0.1 \
  --head_dropout 0.1 \
  --learning_rate 0.0001 \
  --d_model 256 \
  --d_ff 256 \
  --num_workers 1 \
  --channel_mixup \
  --sigma 0.5 \
  --channel_att \
  --avg \
  --max \
  --reduction 2 \
  --itr 1 >> ./log/log_weather_$seq_len'_'$pred_len.txt
done
done
