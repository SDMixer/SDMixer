for pred_len in 96 192 336 720
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../dataset/ \
  --data_path traffic.csv \
  --model_id traffic_${pred_len} \
  --model SDMixer \
  --data custom \
  --enc_in 862 \
  --pred_len ${pred_len} \
  --train_epochs 100 \
  --batch_size 256 \
  --patience 5 \
  --learning_rate 0.001 \
  --des 'Exp' \
  --itr 1 
done