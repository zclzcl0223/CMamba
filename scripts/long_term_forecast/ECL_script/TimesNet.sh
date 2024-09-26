export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./log" ]; then
    mkdir ./log
fi

if [ ! -d "./log/TimesNet" ]; then
    mkdir ./log/TimesNet
fi

if [ ! -d "./log/TimesNet/ecl" ]; then
    mkdir ./log/TimesNet/ecl
fi

model_name=TimesNet

for pred_len in 96 192 336 720
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --d_model 256 \
  --d_ff 512 \
  --top_k 5 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --num_workers 1 \
  --channel_mixup \
  --sigma 0.5 \
  --gddmlp \
  --avg \
  --max \
  --reduction 8 \
  --loss mse \
  --itr 1 | tee -a ./log/TimesNet/ecl/log.txt
done