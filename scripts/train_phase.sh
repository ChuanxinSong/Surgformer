CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch \
--nproc_per_node=3 \
--master_port 12324 \
downstream_phase/run_phase_training.py \
--batch_size 8 \
--epochs 50 \
--save_ckpt_freq 10 \
--model  surgformer_base \
--pretrained_path /home/user/scx/Weight/timesformer/TimeSformer_divST_8x32_224_K400.pyth \
--mixup 0.8 \
--cutmix 1.0 \
--smoothing 0.1 \
--lr 5e-4 \
--layer_decay 0.75 \
--warmup_epochs 5 \
--data_path /media/user/4TB-2/cholec80/surgformer \
--eval_data_path /media/user/4TB-2/cholec80/surgformer \
--nb_classes 7 \
--data_strategy online \
--output_mode key_frame \
--num_frames 16 \
--sampling_rate 4 \
--data_set Cholec80 \
--data_fps 1fps \
--output_dir /home/user/scx/Code/Surgformer/results/cholec80 \
--log_dir /home/user/scx/Code/Surgformer/results/cholec80 \
--num_workers 10 \
--dist_eval \
# --enable_deepspeed \
# --no_auto_resume
