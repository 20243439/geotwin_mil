python main.py \
  --mode few \
  --shot 16  \
  --feature_type image \
  --model_name geotwin \
  --label_dir "./seoul_image/label" \
  --batch_size 32 \
  --output_dim 6 \
  --lr 1e-3 \
  --weight_decay 1e-5 \
  --num_epochs 50 \
  --early stop patience 10 \
  --seed 42

  
  
  
