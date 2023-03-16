4090:
	python3 main.py --batch_size 2048 --learning_rate 0.00002 --num_sanity_val_steps 1 --patience 2 --model ResNet
# python3 main.py --batch_size 2048 --learning_rate 0.00002 --num_sanity_val_steps 1 --patience 2 --model ViT
# python3 main.py --batch_size 64 --learning_rate 0.0002 --num_sanity_val_steps 1 --patience 2 --model YOLO

M2:
	python3 main.py --batch_size 32 --learning_rate 0.00002 --num_sanity_val_steps 1

auto:
	python3 main.py --auto_scale_batch_size true --auto_lr_find true --num_sanity_val_steps 1

fast:
	python3 main.py --fast_dev_run --model ResNet

clean:
	rm -f logs_*.json
