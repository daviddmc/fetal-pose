python main.py --name=b8s64bn_rot_scale0.1_zoom_2_0.6_adamwr1_flip_unet_all\
               --gpu_id=0 \
               --lr=1e-3 \
               --lr_decay_gamma=0.0 \
               --lr_decay_method=cos_restart \
               --optimizer=adamw1e-4 \
               --nStacks=1\
               --batch_size=8\
               --rot\
               --flip\
               --scale=0.1\
               --zoom=2\
               --zoom_factor=0.6\
               --nFeat=64\
               --network=unet\
               --crop_size=56,56,56\
               --epochs=200\
               --lr_decay_ep=201\
               --train_all
               # 200, 13.4