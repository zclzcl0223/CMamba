export CUDA_VISIBLE_DEVICES=0

model_name=PatchTST

if [ ! -d "./log" ]; then
    mkdir ./log
fi

if [ ! -d "./log/PatchTST" ]; then
    mkdir ./log/PatchTST
fi

if [ ! -d "./log/PatchTST/weather" ]; then
    mkdir ./log/PatchTST/weather
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
  --n_heads 4 \
  --train_epochs 100 \
  --channel_mixup \
  --sigma 0.5 \
  --gddmlp \
  --avg \
  --max \
  --reduction 2 \
  --loss mse \
  --itr 1 | tee -a ./log/PatchTST/weather/log.txt
done