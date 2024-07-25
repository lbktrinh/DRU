BATCH_SIZE=8
DATA_ROOT=./data
OUTPUT_DIR=./outputs/def-detr-base/sim2city/teaching_mask/evaluation

CUDA_VISIBLE_DEVICES=0 python -u main.py \
--backbone resnet50 \
--num_encoder_layers 6 \
--num_decoder_layers 6 \
--num_classes 4 \
--data_root ${DATA_ROOT} \
--source_dataset sim10k \
--target_dataset cityscapes \
--eval_batch_size ${BATCH_SIZE} \
--mode eval \
--output_dir ${OUTPUT_DIR} \
--resume ${OUTPUT_DIR}/../sim2city_teaching_mask.pth

