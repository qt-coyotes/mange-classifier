4090:
	python3 main.py --batch_size 16 --learning_rate 0.00001 --num_sanity_val_steps 1 --patience 5 --model YOLO --yolo_model yolov8n-cls.pt --criterion awBCELoss
	python3 main.py --batch_size 16 --learning_rate 0.00001 --num_sanity_val_steps 1 --patience 5 --model YOLO --yolo_model yolov8s-cls.pt --criterion awBCELoss
	python3 main.py --batch_size 16 --learning_rate 0.00001 --num_sanity_val_steps 1 --patience 5 --model YOLO --yolo_model yolov8m-cls.pt --criterion awBCELoss
	python3 main.py --batch_size 16 --learning_rate 0.00001 --num_sanity_val_steps 1 --patience 5 --model YOLO --yolo_model yolov8l-cls.pt  --criterion awBCELoss
	python3 main.py --batch_size 16 --learning_rate 0.00001 --num_sanity_val_steps 1 --patience 5 --model YOLO --yolo_model yolov8x-cls.pt  --criterion awBCELoss

M2:
	python3 main.py --batch_size 32 --learning_rate 0.00002 --num_sanity_val_steps 1

auto:
	python3 main.py --auto_scale_batch_size true --auto_lr_find true --num_sanity_val_steps 1

fast:
	python3 main.py --fast_dev_run --learning_rate 0.00001 --num_sanity_val_steps 1 --patience 5 --model ResNet --resnet_model ResNet34 --criterion awBCELoss --metadata_path data/CHIL/CHIL_uwin_mange_Marit_07242020.json

clean:
	rm -f logs_*.json
	rm -f logs_*.tsv
	rm -rf lightning_logs
