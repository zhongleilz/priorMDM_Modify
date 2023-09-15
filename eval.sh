module load cuda/11.8
module load gcc/11.1.0
module load ffmpeg/20190305
# python -m eval.eval_finetuned_motion_control --model_path save/root_humanml_full_csignal_finetuned/model000220000.pt --replication_times 1 --inpainting_mask root  --overwrite --masked_rate 1
# python -m eval.eval_finetuned_motion_control --model_path save/root_humanml_full_csignal_finetuned/model000220000.pt --replication_times 1 --inpainting_mask root  --overwrite --masked_rate 2
# python -m eval.eval_finetuned_motion_control --model_path save/root_humanml_full_csignal_finetuned/model000220000.pt --replication_times 1 --inpainting_mask root  --overwrite --masked_rate 5
# python -m eval.eval_finetuned_motion_control --model_path save/root_humanml_full_csignal_finetuned/model000220000.pt --replication_times 1 --inpainting_mask root  --overwrite --masked_rate 25
# python -m eval.eval_finetuned_motion_control --model_path save/root_humanml_full_csignal_finetuned/model000220000.pt --replication_times 1 --inpainting_mask root  --overwrite --masked_rate 100
python -m eval.eval_finetuned_motion_control --model_path save/root_kit_25%_csignal_finetuned/model000480000.pt --replication_times 1 --inpainting_mask root  --overwrite --masked_rate 25 --dataset kit
# python -m eval.eval_finetuned_motion_control --model_path save/root_kit_full_csignal_finetuned/model000420000.pt --replication_times 1 --inpainting_mask root  --overwrite --masked_rate 2 --dataset kit
# python -m eval.eval_finetuned_motion_control --model_path save/root_kit_full_csignal_finetuned/model000420000.pt --replication_times 1 --inpainting_mask root  --overwrite --masked_rate 5 --dataset kit
# python -m eval.eval_finetuned_motion_control --model_path save/root_kit_full_csignal_finetuned/model000420000.pt --replication_times 1 --inpainting_mask root  --overwrite --masked_rate 25 --dataset kit
# python -m eval.eval_finetuned_motion_control --model_path save/root_kit_full_csignal_finetuned/model000420000.pt --replication_times 1 --inpainting_mask root  --overwrite --masked_rate 100 --dataset kit