export CUDA_VISIBLE_DEVICES=0

model_name=PatchTST

if [ ! -d "./log" ]; then
    mkdir ./log
fi

if [ ! -d "./log/PatchTST" ]; then
    mkdir ./log/PatchTST
fi

if [ ! -d "./log/PatchTST/ecl" ]; then
    mkdir ./log/PatchTST/ecl
fi


for pred_len in 96 192 336
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
  --des 'Exp' \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --train_epochs 100 \
  --num_workers 1 \
  --channel_mixup \
  --sigma 1.0 \
  --gddmlp \
  --avg \
  --max \
  --reduction 4 \
  --loss mse \
  --itr 1 | tee -a ./log/PatchTST/ecl/log.txt
done

for pred_len in 720
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
  --des 'Exp' \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --train_epochs 100 \
  --num_workers 1 \
  --channel_mixup \
  --sigma 1.0 \
  --gddmlp \
  --avg \
  --max \
  --reduction 4 \
  --loss mse \
  --itr 1 | tee -a ./log/PatchTST/ecl/log.txt
done
