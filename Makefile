4090:
# python3 main.py --batch_size 512 --learning_rate 0.00002 --num_sanity_val_steps 1 --patience 5 --model ResNet
	python3 main.py --batch_size 64 --learning_rate 0.00002 --num_sanity_val_steps 1 --patience 5 --model DenseNet
# python3 main.py --batch_size 64 --learning_rate 0.00002 --num_sanity_val_steps 1 --patience 5 --model ViT
# python3 main.py --batch_size 64 --learning_rate 0.0002 --num_sanity_val_steps 1 --patience 5 --model YOLO --yolo_model yolov8s-cls --max_epochs 100

M2:
	python3 main.py --batch_size 32 --learning_rate 0.00002 --num_sanity_val_steps 1

auto:
	python3 main.py --auto_scale_batch_size true --auto_lr_find true --num_sanity_val_steps 1

fast:
	python3 main.py --fast_dev_run --model DenseNet

clean:
	rm -f logs_*.json
	rm -rf lightning_logs
