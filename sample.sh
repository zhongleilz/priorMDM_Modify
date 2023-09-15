# conda /work/vig/zhonglei/anaconda/envs/mdm
module load cuda/11.8
module load gcc/11.1.0
module load ffmpeg/20190305
clear
python -m sample.finetuned_motion_control --model_path save/root_kit_sparse_csignal_finetuned/model000600000.pt --dataset kit --show_input --inpainting_mask root --masked_rate 25