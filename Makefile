4090:
	python3 main.py --batch_size 64 --learning_rate 0.001 --num_sanity_val_steps 1 --patience 10 --model ResNet --criterion BCELoss --internal_k 10 --num_workers 12 --scheduler_patience 8
# python3 main.py --batch_size 32 --learning_rate 0.00001 --num_sanity_val_steps 1 --patience 10 --model YOLO --yolo_model yolov8s-cls.pt --criterion ExpectedCostLoss --internal_k 10 --num_workers 12 --scheduler_patience 8

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
