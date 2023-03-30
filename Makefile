4090:
	python3 main.py --batch_size 32 --learning_rate 0.0001 --num_sanity_val_steps 1 --patience 5 --model ResNet --criterion ExpectedCostLoss --internal_k 10 --num_workers 12 --scheduler_patience 4 --no_external_group --internal_group

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
