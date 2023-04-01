4090:
	python3 main.py --batch_size 16 --learning_rate 0.00001 --num_sanity_val_steps 1 --patience 5 --model Densenet --densenet_model Densenet121 --criterion wBCELoss --criterion_pos_weight 8
	python3 main.py --batch_size 16 --learning_rate 0.00001 --num_sanity_val_steps 1 --patience 5 --model Densenet --densenet_model DenseNet161 --criterion wBCELoss --criterion_pos_weight 8
	python3 main.py --batch_size 16 --learning_rate 0.00001 --num_sanity_val_steps 1 --patience 5 --model Densenet --densenet_model DenseNet169 --criterion wBCELoss --criterion_pos_weight 8
	python3 main.py --batch_size 16 --learning_rate 0.00001 --num_sanity_val_steps 1 --patience 5 --model Densenet --densenet_model DenseNet201 --criterion wBCELoss --criterion_pos_weight 8
	python3 main.py --batch_size 16 --learning_rate 0.00001 --num_sanity_val_steps 1 --patience 5 --model Densenet --densenet_model Densenet121 --criterion wBCELoss --criterion_pos_weight 7
	python3 main.py --batch_size 16 --learning_rate 0.00001 --num_sanity_val_steps 1 --patience 5 --model Densenet --densenet_model DenseNet161 --criterion wBCELoss --criterion_pos_weight 7
	python3 main.py --batch_size 16 --learning_rate 0.00001 --num_sanity_val_steps 1 --patience 5 --model Densenet --densenet_model DenseNet169 --criterion wBCELoss --criterion_pos_weight 7
	python3 main.py --batch_size 16 --learning_rate 0.00001 --num_sanity_val_steps 1 --patience 5 --model Densenet --densenet_model DenseNet201 --criterion wBCELoss --criterion_pos_weight 7
	python3 main.py --batch_size 16 --learning_rate 0.00001 --num_sanity_val_steps 1 --patience 5 --model Densenet --densenet_model Densenet121 --criterion wBCELoss --criterion_pos_weight 6
	python3 main.py --batch_size 16 --learning_rate 0.00001 --num_sanity_val_steps 1 --patience 5 --model Densenet --densenet_model DenseNet161 --criterion wBCELoss --criterion_pos_weight 6
	python3 main.py --batch_size 16 --learning_rate 0.00001 --num_sanity_val_steps 1 --patience 5 --model Densenet --densenet_model DenseNet169 --criterion wBCELoss --criterion_pos_weight 6
	python3 main.py --batch_size 16 --learning_rate 0.00001 --num_sanity_val_steps 1 --patience 5 --model Densenet --densenet_model DenseNet201 --criterion wBCELoss --criterion_pos_weight 6
	python3 main.py --batch_size 16 --learning_rate 0.00001 --num_sanity_val_steps 1 --patience 5 --model Densenet --densenet_model Densenet121 --criterion wBCELoss --criterion_pos_weight 4
	python3 main.py --batch_size 16 --learning_rate 0.00001 --num_sanity_val_steps 1 --patience 5 --model Densenet --densenet_model DenseNet161 --criterion wBCELoss --criterion_pos_weight 4
	python3 main.py --batch_size 16 --learning_rate 0.00001 --num_sanity_val_steps 1 --patience 5 --model Densenet --densenet_model DenseNet169 --criterion wBCELoss --criterion_pos_weight 4
	python3 main.py --batch_size 16 --learning_rate 0.00001 --num_sanity_val_steps 1 --patience 5 --model Densenet --densenet_model DenseNet201 --criterion wBCELoss --criterion_pos_weight 4
	python3 main.py --batch_size 16 --learning_rate 0.00001 --num_sanity_val_steps 1 --patience 5 --model Densenet --densenet_model Densenet121 --criterion wBCELoss --criterion_pos_weight 3
	python3 main.py --batch_size 16 --learning_rate 0.00001 --num_sanity_val_steps 1 --patience 5 --model Densenet --densenet_model DenseNet161 --criterion wBCELoss --criterion_pos_weight 3
	python3 main.py --batch_size 16 --learning_rate 0.00001 --num_sanity_val_steps 1 --patience 5 --model Densenet --densenet_model DenseNet169 --criterion wBCELoss --criterion_pos_weight 3
	python3 main.py --batch_size 16 --learning_rate 0.00001 --num_sanity_val_steps 1 --patience 5 --model Densenet --densenet_model DenseNet201 --criterion wBCELoss --criterion_pos_weight 3
	python3 main.py --batch_size 16 --learning_rate 0.00001 --num_sanity_val_steps 1 --patience 5 --model Densenet --densenet_model Densenet121 --criterion wBCELoss --criterion_pos_weight 2
	python3 main.py --batch_size 16 --learning_rate 0.00001 --num_sanity_val_steps 1 --patience 5 --model Densenet --densenet_model DenseNet161 --criterion wBCELoss --criterion_pos_weight 2
	python3 main.py --batch_size 16 --learning_rate 0.00001 --num_sanity_val_steps 1 --patience 5 --model Densenet --densenet_model DenseNet169 --criterion wBCELoss --criterion_pos_weight 2
	python3 main.py --batch_size 16 --learning_rate 0.00001 --num_sanity_val_steps 1 --patience 5 --model Densenet --densenet_model DenseNet201 --criterion wBCELoss --criterion_pos_weight 2

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
