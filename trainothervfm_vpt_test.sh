gpu=0
data=ham10000
noiserate=0.4

echo ${gpu} ${data} ${noiserate}
python train_rein_other_vfm.py -d ham10000 -g 0 -s dinov1b_vpt_0.0 -n 0.0 --net dinov1 --adapter vpt
# python train_rein_other_vfm_ours_three_head.py -d ham10000 -g 0 -s dinov1b_rein_ours_threehead_0.4 -n 0.4 --net dinov1 --adapter adaptformer

python train_rein_other_vfm.py -d ham10000 -g 0 -s maeb_vpt_0.0 -n 0.0 --net mae --adapter vpt
# python train_rein_other_vfm_ours_three_head.py -d ham10000 -g 0 -s maeb_rein_ours_threehead_0.4 -n 0.4 --net mae --adapter adaptformer

python train_rein_other_vfm.py -d ham10000 -g 0 -s dinov2b_vpt_0.0 -n 0.0 --net dinov2 --adapter vpt
# python train_rein_other_vfm_ours_three_head.py -d ham10000 -g 0 -s dinov2b_rein_ours_threehead_0.4 -n 0.4 --net dinov2 --adapter adaptformer