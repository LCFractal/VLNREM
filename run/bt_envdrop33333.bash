name=agent_bt3333
# aug: the augmented paths, only the paths are used (not the insts)
# speaker: load the speaker from
# load: load the agent from
flag="--attn soft --train auglistener --selfTrain
      --aug tasks/R2R/data/aug_paths.json
      --speaker snap/speaker/state_dict/best_val_unseen_bleu 
      --load snap/best_mod/agent33333/state_dict/best_val_unseen
      --angleFeatSize 128
      --accumulateGrad
      --featdropout 0.4
      --subout max --optim rms --lr 1e-4 --iters 800000 --maxAction 35"
mkdir -p snap/$name
# CUDA_VISIBLE_DEVICES=$1 python r2r_src/train.py $flag --name $name 

# Try this with file logging:
CUDA_VISIBLE_DEVICES=$1 unbuffer python r2r_src/train.py $flag --name $name | tee snap/$name/log
