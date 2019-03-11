python ~/Documents/distributed_platform/classification.py --data ~/Documents/data --num_inst 4 --inst_id 1 --train --val --no-test --model linear --saved_model_name linear_model --log log1.txt --img_height 256 --img_width 256 --img_channels 3 --num_classes 2 --max_cycles 200 --sleep_time 30

tar xvzf linear_model_best.tar.gz

python ~/Documents/distributed_platform/classification.py --data ~/Documents/data --num_inst 4 --inst_id 1 --no-train --no-val --test --model linear --load linear_model --log log1.txt --img_height 256 --img_width 256 --img_channels 3 --num_classes 2 --sleep_time 30

git credential-cache exit