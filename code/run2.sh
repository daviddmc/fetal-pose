python main2.py --name=b5s64bn_rot_scale0.1_adamwr1_newtest_flip_unet_all \
               --ngpu=1 \
               --scale=0.1 \
               --epochs=200 \
               --bone=[[11,5],[12,6],[5,13],[6,14],[9,2],[10,3],[2,0],[3,1]] \
               --boneLambda=0 \
               --lr=5e-3 \
               --lr_decay_ep=13.4 \
               --lr_decay_gamma=0.0 \
               --lr_decay_method=cos_restart \
               --optimizer=adamw1e-4 \
               --nStacks=1 \
               --batch_size=5\
               --dataset="[('040716', 300, 1), ('043015', 400, 1), ('031616', 100, 1), ('031615', 100, 1), ('022618', 100, 1), ('102617', 100, 1), ('111', 64, 5), ('040218', 70, 1), ('032318a', 70, 1), ('061217', 20, 1)]"\
               --rot \
               --flip \
               --run=test \
               --network=unet \
               --nFeat=64 \
               #--train_like_test=0.25 \
               #--temporal \
               # 6.5 13.4
