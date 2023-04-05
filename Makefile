4090:
	python3 main.py --batch_size 32 --auto_lr_find --num_sanity_val_steps 1 --patience 5 --model ResNet --resnet_model ResNet18 --criterion awBCELoss --monitor val_loss --nonpretrained --no_crop --no_data_augmentation --no_tabular_features --message "ResNet18 with awBCELoss"

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
