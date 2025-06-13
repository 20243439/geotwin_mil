python main.py --model_name resnet50 --output_dim 22 --feature_type map --mode few --shot 1

python main.py --model_name vit_b_16

python main.py --model_name swin_v2_t

python main.py --model_name simclr 

python main.py --model_name min

python main.py --model_name mean 

python main.py --model_name max

python main.py --model_name hadamard

python main.py --model_name conc

python main.py --model_name gated

python main.py --model_name cross

python main.py --model_name moco

python warm.py --model_name moco --output_dim 22 --train_group small --valid_group small --test_group small --seed 1

python cold.py \
  --image_dir "./beijing_image/image" \
  --label_dir "./beijing_image/label" \
  --checkpoint best_resnet_func_1.pth \
  --output_json bj_resnet_func_1.json \
  --model_name resnet50 \
  --output_dim 6 \
  --patch_size 224
