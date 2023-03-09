4090:
	python3 main.py --batch_size 64 --learning_rate 0.00002 --num_sanity_val_steps 1

auto:
	python3 main.py --auto_scale_batch_size true --auto_lr_find true --num_sanity_val_steps 1

fast:
	python3 main.py --fast_dev_run
