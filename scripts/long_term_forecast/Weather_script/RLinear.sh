export CUDA_VISIBLE_DEVICES=0

model_name=RLinear

if [ ! -d "./log" ]; then
    mkdir ./log
fi

if [ ! -d "./log/RLinear" ]; then
    mkdir ./log/RLinear
fi

if [ ! -d "./log/RLinear/weather" ]; then
    mkdir ./log/RLinear/weather
fi

for pred_len in 96 192 336 720
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
  --des 'Exp' \
  --batch_size 128 \
  --learning_rate 0.005 \
  --rev \
  --train_epochs 100 \
  --num_workers 1 \
  --channel_mixup \
  --sigma 0.1 \
  --gddmlp \
  --avg \
  --max \
  --reduction 2 \
  --loss mse \
  --itr 1 | tee -a ./log/RLinear/weather/log.txt
done
