for pred_len in 96 192 336 720
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_${pred_len} \
  --model SDMixer \
  --data ETTh2 \
  --enc_in 7 \
  --pred_len ${pred_len} \
  --train_epochs 15 \
  --batch_size 256 \
  --patience 5 \
  --learning_rate 0.001 \
  --des 'Exp' \
  --itr 1 
done