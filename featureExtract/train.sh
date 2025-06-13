python main.py \
  --root_dir "./seoul_image/patch_images" \
  --map_dir "./seoul_map/patch_images" \
  --mode train \
  --batch_size 32 \
  --epochs 50 \
  --lr 1e-3 \
  --rank 4 \
  --pretrained_path "./featureExtract/r-50-1000ep.pth.tar" \
  --save_path "moco_model_512_re.pth"

python main.py \
  --mode extract --model_path "moco_model_512.pth" \
  --root_dir "./seoul_image/patch_images"