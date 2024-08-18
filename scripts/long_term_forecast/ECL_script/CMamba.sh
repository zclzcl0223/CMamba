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
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --lradj type3 \
  --train_epochs 100 \
  --patience 3 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 5 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --batch_size 8 \
  --pscan \
  --head_dropout 0.0 \
  --dropout 0.0 \
  --num_workers 1 \
  --learning_rate 0.001 \
  --channel_mixup \
  --sigma 1.0 \
  --channel_att \
  --avg \
  --max \
  --reduction 4 \
  --itr 1 >> ./log/log_ecl_$seq_len'_'$pred_len.txt
done
done


for seq_len in 96
do
for pred_len in 192
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --lradj type3 \
  --train_epochs 100 \
  --patience 3 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 5 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --batch_size 8 \
  --pscan \
  --head_dropout 0.0 \
  --dropout 0.0 \
  --num_workers 1 \
  --learning_rate 0.001 \
  --channel_mixup \
  --sigma 1.0 \
  --channel_att \
  --avg \
  --max \
  --reduction 4 \
  --itr 1 >> ./log/log_ecl_$seq_len'_'$pred_len.txt
done
done


for seq_len in 96
do
for pred_len in 336
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --lradj type3 \
  --train_epochs 100 \
  --patience 5 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 5 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --batch_size 4 \
  --pscan \
  --head_dropout 0.0 \
  --dropout 0.0 \
  --num_workers 1 \
  --learning_rate 0.0005 \
  --channel_mixup \
  --sigma 1.0 \
  --channel_att \
  --avg \
  --max \
  --reduction 4 \
  --itr 1 >> ./log/log_ecl_$seq_len'_'$pred_len.txt
done
done


for seq_len in 96
do
for pred_len in 720
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --lradj type3 \
  --train_epochs 100 \
  --patience 3 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 5 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --batch_size 4 \
  --pscan \
  --head_dropout 0.0 \
  --dropout 0.0 \
  --num_workers 1 \
  --learning_rate 0.0001 \
  --channel_mixup \
  --sigma 1.0 \
  --channel_att \
  --avg \
  --max \
  --reduction 4 \
  --itr 1 >> ./log/log_ecl_$seq_len'_'$pred_len.txt
done
done
