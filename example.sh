python ~/Documents/dist_plat/distributed_platform/classification.py --central_path exp1 --data ~/Documents/dist_plat/distributed_platform/data1 --num_inst 4 --inst_id 1 --train --val --no-test --model linear --saved_model_name linear_model --log log1.txt --img_height 16 --img_width 16 --img_channels 3 --num_classes 2 --max_cycles 20 --sleep_time 5

tar xvzf linear_model_best.tar.gz

python ~/Documents/dist_plat/distributed_platform/classification.py --central_path exp1 --data ~/Documents/dist_plat/distributed_platform/data1 --num_inst 4 --inst_id 1 --no-train --no-val --test --model linear --load linear_model --log log1.txt --img_height 16 --img_width 16 --img_channels 3 --num_classes 2 --sleep_time 5