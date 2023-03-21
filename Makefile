4090:
	python3 main.py --batch_size 64 --learning_rate 0.0002 --num_sanity_val_steps 1 --patience 5 --model YOLO --yolo_model yolov8s-cls --max_epochs 30 --criterion BCEWithLogitsLoss
# python3 main.py --batch_size 64 --learning_rate 0.00002 --num_sanity_val_steps 1 --patience 5 --model BiT --criterion BCEWithLogitsLoss
# python3 main.py --batch_size 32 --learning_rate 0.0001 --num_sanity_val_steps 1 --max_epochs 20 --min_epochs 20 --model DenseNet
# python3 main.py --batch_size 64 --learning_rate 0.00002 --num_sanity_val_steps 1 --patience 5 --model ViT

M2:
	python3 main.py --batch_size 32 --learning_rate 0.00002 --num_sanity_val_steps 1

auto:
	python3 main.py --auto_scale_batch_size true --auto_lr_find true --num_sanity_val_steps 1

fast:
	python3 main.py --fast_dev_run --model ResNet

clean:
	rm -f logs_*.json
	rm -f logs_*.tsv
	rm -rf lightning_logs
