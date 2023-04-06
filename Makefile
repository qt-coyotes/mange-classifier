4090:
	-python3 main.py --batch_size 32 --auto_lr_find --num_sanity_val_steps 1 --patience 5 --model YOLO --resnet_model yolov8n-cls.pt --criterion awBCELoss --monitor val_loss --no_crop --no_data_augmentation --message "yolov8n-cls pretrained with awBCELoss."
	-python3 main.py --batch_size 32 --auto_lr_find --num_sanity_val_steps 1 --patience 5 --model YOLO --resnet_model yolov8s-cls.pt --criterion awBCELoss --monitor val_loss --no_crop --no_data_augmentation --message "yolov8s-cls pretrained with awBCELoss."
	-python3 main.py --batch_size 32 --auto_lr_find --num_sanity_val_steps 1 --patience 5 --model YOLO --resnet_model yolov8m-cls.pt --criterion awBCELoss --monitor val_loss --no_crop --no_data_augmentation --message "yolov8m pretrained with awBCELoss."
	-python3 main.py --batch_size 32 --auto_lr_find --num_sanity_val_steps 1 --patience 5 --model YOLO --resnet_model yolov8l-cls.pt --criterion awBCELoss --monitor val_loss --no_crop --no_data_augmentation --message "yolov8l pretrained with awBCELoss."
	-python3 main.py --batch_size 32 --auto_lr_find --num_sanity_val_steps 1 --patience 5 --model YOLO --resnet_model yolov8x-cls.pt --criterion awBCELoss --monitor val_loss --no_crop --no_data_augmentation --message "yolov8x pretrained with awBCELoss."


CHIL:
	python3 main.py --batch_size 16 --auto_lr_find --num_sanity_val_steps 1 --patience 5 --model ResNet --resnet_model ResNet18 --criterion BCELoss --monitor val_loss --no_crop --no_tabular_features --metadata_path data/CHIL/CHIL_uwin_mange_Marit_07242020.json --message "pretrained + data augmentation"
	python3 main.py --batch_size 16 --auto_lr_find --num_sanity_val_steps 1 --patience 5 --model ResNet --resnet_model ResNet18 --criterion BCELoss --monitor val_loss --no_tabular_features --metadata_path data/CHIL/CHIL_uwin_mange_Marit_07242020.json --message "pretrained + crop + data augmentation"

M2:
	python3 main.py --batch_size 16 --learning_rate 0.0001 --num_sanity_val_steps 1 --patience 5 --model Random

auto:
	python3 main.py --auto_scale_batch_size true --auto_lr_find true --num_sanity_val_steps 1

fast:
	python3 main.py --fast_dev_run --learning_rate 0.00001 --num_sanity_val_steps 1 --patience 5 --model ResNet --resnet_model ResNet34 --criterion awBCELoss --metadata_path data/CHIL/CHIL_uwin_mange_Marit_07242020.json --no_tabular_features

clean:
	rm -f logs_*.json
	rm -f logs*.tsv
	rm -rf lightning_logs
