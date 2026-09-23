# python test.py --dataset_root /datasets/graspnet --camera realsense --checkpoint_path logs/gsnet_pointcept_pretrained_lower_backbone_lr/gsnet_pointcept_epoch27.tar --dump_dir dumps/realsense_backbone_e21 --batch_size 1 --infer --eval --backbone transformer_pretrained 
python test.py \
--dataset_root ../graspnet \
--camera realsense \
--checkpoint_path logs/gsnet_dev/gsnet_dev_epoch02.tar \
--dump_dir dumps/realsense_dev \
--batch_size 1 \
--infer \
--eval \
--backbone resunet \
--num_point 15000 \
--split test_seen_single