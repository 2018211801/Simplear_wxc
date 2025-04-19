cd /openseg_blob/wxc/SimpleAR
accelerate launch --main_process_port 1234  --num_processes 3 --config_file  simpar/configs/accelerate_configs/zero3.yaml \
    --num_processes=4 simpar/train/llava_trainer_grpo.py \
    --config simpar/configs/config_grpo.yaml \
    --data_path /openseg_blob/wxc/SimpleAR/datasets/one_animal_grid_layout_5000_refine_v03_metadata_nolist2.json