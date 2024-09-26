export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./log" ]; then
    mkdir ./log
fi

if [ ! -d "./log/iTransformer" ]; then
    mkdir ./log/iTransformer
fi

if [ ! -d "./log/iTransformer/weather" ]; then
    mkdir ./log/iTransformer/weather
fi

model_name=iTransformer

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
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --train_epochs 100 \
  --num_workers 1 \
  --channel_mixup \
  --sigma 0.5 \
  --gddmlp \
  --avg \
  --max \
  --loss mse \
  --itr 1 | tee -a ./log/iTransformer/weather/log.txt
done
