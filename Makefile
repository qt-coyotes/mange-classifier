4090:
	python3 main.py --batch_size 16 --learning_rate 0.001 --num_sanity_val_steps 1 --patience 5 --model ResNet --resnet_model ResNet18 --nopretrained --criterion ExpectedCostLoss --criterion_cfn 5--metadata_path data/CHIL/CHIL_uwin_mange_Marit_07242020.json
	python3 main.py --batch_size 16 --learning_rate 0.001 --num_sanity_val_steps 1 --patience 5 --model ResNet --resnet_model ResNet34 --nopretrained --criterion ExpectedCostLoss --criterion_cfn 5--metadata_path data/CHIL/CHIL_uwin_mange_Marit_07242020.json
	python3 main.py --batch_size 16 --learning_rate 0.001 --num_sanity_val_steps 1 --patience 5 --model ResNet --resnet_model ResNet50 --nopretrained --criterion ExpectedCostLoss --criterion_cfn 5--metadata_path data/CHIL/CHIL_uwin_mange_Marit_07242020.json
	python3 main.py --batch_size 16 --learning_rate 0.001 --num_sanity_val_steps 1 --patience 5 --model ResNet --resnet_model ResNet101 --nopretrained --criterion ExpectedCostLoss --criterion_cfn 5--metadata_path data/CHIL/CHIL_uwin_mange_Marit_07242020.json
	python3 main.py --batch_size 16 --learning_rate 0.001 --num_sanity_val_steps 1 --patience 5 --model ResNet --resnet_model ResNet152 --nopretrained --criterion ExpectedCostLoss --criterion_cfn 5--metadata_path data/CHIL/CHIL_uwin_mange_Marit_07242020.json

M2:
	python3 main.py --batch_size 32 --learning_rate 0.00002 --num_sanity_val_steps 1

auto:
	python3 main.py --auto_scale_batch_size true --auto_lr_find true --num_sanity_val_steps 1

fast:
	python3 main.py --fast_dev_run --learning_rate 0.00002 --num_sanity_val_steps 1 --patience 5 --model ResNet --criterion wBCELoss --criterion_pos_weight 10 --metadata_path data/CHIL/CHIL_uwin_mange_Marit_07242020.json

clean:
	rm -f logs_*.json
	rm -f logs_*.tsv
	rm -rf lightning_logs
