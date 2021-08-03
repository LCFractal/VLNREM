name=eval_agent
flag="--attn    soft      --train   validlistener --selfTrain
      --aug     tasks/R2R/data/aug_paths.json
      --speaker snap/speaker/state_dict/best_val_unseen_bleu
      --load    snap/agent_bt/state_dict/best_val_unseen
      --angleFeatSize 128 --accumulateGrad
      --featdropout 0.4   --subout  max           --optim   rms
      --lr      1e-4      --iters   200000        --maxAction 35"

CUDA_VISIBLE_DEVICES=$1 python r2r_src/train.py $flag --name $name 

# Try this with file logging:
# CUDA_VISIBLE_DEVICES=$1 unbuffer python r2r_src/train.py $flag --name $name | tee snap/$name/log
