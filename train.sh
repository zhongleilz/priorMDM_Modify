module load cuda/11.8
module load gcc/11.1.0
module load ffmpeg/20190305

python -m train.train_mdm_motion_control --save_dir save/root_humanml_sparse_csignal_finetuned --resume_checkpoint save/humanml_trans_enc_512/model000200000.pt --inpainting_mask root --save_interval 10_000 --overwrite --masked_rate 66
# python -m train.train_mdm_motion_control --save_dir save/root_full_csignal_finetuned_kit --resume_checkpoint save/kit_trans_enc_512/model000400000.pt --inpainting_mask root --save_interval 10_000 --overwrite --dataset kit
