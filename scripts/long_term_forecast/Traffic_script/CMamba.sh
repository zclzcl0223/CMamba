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
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 4 \
  --lradj type3 \
  --train_epochs 100 \
  --patience 5 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --d_model 128 \
  --d_ff 128 \
  --des 'Exp' \
  --batch_size 8 \
  --num_workers 1 \
  --pscan \
  --head_dropout 0.0 \
  --dropout 0.0 \
  --learning_rate 0.001 \
  --channel_mixup \
  --sigma 4.0 \
  --channel_att \
  --avg \
  --max \
  --reduction 8 \
  --itr 1 >> ./log/log_traffic_$seq_len'_'$pred_len.txt
done
done


for seq_len in 96
do
for pred_len in 192
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 4 \
  --lradj type3 \
  --train_epochs 100 \
  --patience 5 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --d_model 128 \
  --d_ff 128 \
  --des 'Exp' \
  --batch_size 4 \
  --num_workers 1 \
  --pscan \
  --head_dropout 0.0 \
  --dropout 0.0 \
  --learning_rate 0.001 \
  --channel_mixup \
  --sigma 1.0 \
  --channel_att \
  --avg \
  --max \
  --reduction 8 \
  --itr 1 >> ./log/log_traffic_$seq_len'_'$pred_len.txt
done
done


for seq_len in 96
do
for pred_len in 336
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 4 \
  --lradj type3 \
  --train_epochs 100 \
  --patience 5 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --d_model 128 \
  --d_ff 128 \
  --des 'Exp' \
  --batch_size 4 \
  --num_workers 1 \
  --pscan \
  --head_dropout 0.0 \
  --dropout 0.0 \
  --learning_rate 0.001 \
  --channel_mixup \
  --sigma 3.0 \
  --channel_att \
  --avg \
  --max \
  --reduction 8 \
  --itr 1 >> ./log/log_traffic_$seq_len'_'$pred_len.txt
done
done


for seq_len in 96
do
for pred_len in 720
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 4 \
  --d_layers 1 \
  --lradj type3 \
  --train_epochs 100 \
  --patience 5 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --d_model 128 \
  --d_ff 128 \
  --des 'Exp' \
  --batch_size 2 \
  --num_workers 1 \
  --pscan \
  --head_dropout 0.0 \
  --dropout 0.0 \
  --learning_rate 0.001 \
  --channel_mixup \
  --sigma 1.0 \
  --channel_att \
  --avg \
  --max \
  --reduction 8 \
  --itr 1 >> ./log/log_traffic_$seq_len'_'$pred_len.txt
done
done
