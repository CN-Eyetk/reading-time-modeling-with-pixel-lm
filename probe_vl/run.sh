#get gpt2 and mgpt surprisal
CUDA_VISIBLE_DEVICES=0 python gen_rm.py
#get bert surprisal
CUDA_VISIBLE_DEVICES=0 python gen_bert.py
#get pixel surprisal
CUDA_VISIBLE_DEVICES=0 python3 gen_plm.py --is_m4 --pixel_lm_right_blur
CUDA_VISIBLE_DEVICES=0 python3 gen_plm.py --is_m4 --pixel_lm_right_blur  --pixel_whole_ia_masking
CUDA_VISIBLE_DEVICES=0 python3 gen_plm.py --is_m4 --pixel_whole_ia_masking
CUDA_VISIBLE_DEVICES=0 python3 gen_plm.py --is_m4 
