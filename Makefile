4090:
	python3 main.py --batch_size 32 --learning_rate 0.00002 --num_sanity_val_steps 1 --patience 5 --model ResNet --criterion BCELoss --metadata_path data/CHIL/CHIL_uwin_mange_Marit_07242020.json --path_prefix CHIL/

M2:
	python3 main.py --batch_size 32 --learning_rate 0.00002 --num_sanity_val_steps 1

auto:
	python3 main.py --auto_scale_batch_size true --auto_lr_find true --num_sanity_val_steps 1

fast:
	python3 main.py --fast_dev_run --learning_rate 0.00002 --num_sanity_val_steps 1 --patience 5 --model ResNet --criterion BCELoss --metadata_path data/CHIL/CHIL_uwin_mange_Marit_07242020.json --path_prefix CHIL/

clean:
	rm -f logs_*.json
	rm -f logs_*.tsv
	rm -rf lightning_logs
