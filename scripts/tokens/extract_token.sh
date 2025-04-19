torchrun \
 --nnodes=1 --nproc_per_node=4 --master_port 2328 \
 /openseg_blob/wxc/SimpleAR/simpar/data/extract_token.py \
    --dataset_type "image" \
    --dataset_name "example" \
    --code_path /openseg_blob/wxc/SimpleAR/datasets/visual_tokens \
    --gen_data_path /openseg_blob/wxc/SimpleAR/datasets/one_animal_grid_layout_5000_refine_v03_metadata.json \
    --gen_image_folder "" \
    --gen_resolution 1024
