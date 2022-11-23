python main.py --name=b8s64bn_rot_scale0.2_zoom_0.5_1.5_adamwr1_flip_unet_all_400ep_new \
               --run=train \
               --gpu_id=6 \
               --lr=1e-3 \
               --lr_decay_gamma=0.0 \
               --lr_decay_method=cos_restart \
               --optimizer=adamw1e-4 \
               --nStacks=1 \
               --batch_size=8 \
               --rot \
               --flip \
               --scale=0.2 \
               --zoom=0.5 \
               --zoom_factor=1.5 \
               --nFeat=64 \
               --network=unet \
               --crop_size=64,64,64 \
               --epochs=400 \
               --lr_decay_ep=401 \
               --norm \
               --train_all