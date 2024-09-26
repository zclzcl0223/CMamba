export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./log" ]; then
    mkdir ./log
fi

if [ ! -d "./log/TimesNet" ]; then
    mkdir ./log/TimesNet
fi

if [ ! -d "./log/TimesNet/weather" ]; then
    mkdir ./log/TimesNet/weather
fi

model_name=TimesNet

for pred_len in 96
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 32 \
  --d_ff 32 \
  --top_k 5 \
  --des 'Exp' \
  --learning_rate 0.0005 \
  --train_epochs 100 \
  --num_workers 1 \
  --channel_mixup \
  --sigma 0.1 \
  --gddmlp \
  --avg \
  --max \
  --reduction 2 \
  --loss mse \
  --itr 1 | tee -a ./log/TimesNet/weather/log.txt
done

for pred_len in 192
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 32 \
  --d_ff 32 \
  --top_k 5 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --train_epochs 1 \
  --num_workers 1 \
  --channel_mixup \
  --sigma 0.1 \
  --gddmlp \
  --avg \
  --max \
  --reduction 4 \
  --loss mse \
  --itr 1 | tee -a ./log/TimesNet/weather/log.txt
done

for pred_len in 336
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 32 \
  --d_ff 32 \
  --top_k 5 \
  --des 'Exp' \
  --learning_rate 0.0001 \
  --train_epochs 1 \
  --num_workers 1 \
  --channel_mixup \
  --sigma 0.1 \
  --gddmlp \
  --avg \
  --max \
  --reduction 4 \
  --loss mse \
  --itr 1 | tee -a ./log/TimesNet/weather/log.txt
done

for pred_len in 720
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 32 \
  --d_ff 32 \
  --top_k 5 \
  --des 'Exp' \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --num_workers 1 \
  --channel_mixup \
  --sigma 0.1 \
  --gddmlp \
  --avg \
  --max \
  --reduction 2 \
  --loss mse \
  --itr 1 | tee -a ./log/TimesNet/weather/log.txt
done