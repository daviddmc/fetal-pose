python main.py --name=b5s64bn_rot_scale0.1_adamwr1_022618_basic \
               --ngpu=1 \
               --rot \
               --scale=0.1 \
	             --scale_type=mul \
               --epochs=200 \
               --bone=[[11,5],[12,6],[5,13],[6,14],[9,2],[10,3],[2,0],[3,1]] \
               --boneLambda=0 \
               --lr=5e-3 \
               --lr_decay_ep=13.4 \
               --lr_decay_gamma=0.0 \
               --lr_decay_method=cos_restart \
               --optimizer=adamw1e-4 \
               --data_test=022618 \
               --network=simple \
               --depth=3 \
               --nFeat=32 \
               #--train_like_test=0.25 \
               #--temporal \
               #--bone=[[11,5],[12,6],[5,13],[6,14],[9,2],[10,3],[2,0],[3,1]] \
               #--bone=[[9,2],[10,3],[2,0],[3,1]] \
               # 6.5 13.4
               # '031616', '102617', '031615', '022618'
