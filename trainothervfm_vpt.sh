gpu=0
data=ham10000
noiserate=0.4

echo ${gpu} ${data} ${noiserate}
# python train_rein_other_vfm.py -d aptos -g 0 -s maeb_adaptformer_0.4 -n 0.4 --net dinov1 --adapter adaptformer

python train_rein_other_vfm.py -d aptos -g 0 -s dinov1b_vpt_0.4 -n 0.4 --net dinov1 --adapter vpt
python train_rein_other_vfm_ours_three_head.py -d aptos -g 0 -s dinov1b_vpt_ours_threehead_0.4_10 -n 0.4 --net dinov1 --adapter vpt

python train_rein_other_vfm.py -d aptos -g 0 -s maeb_vpt_0.4 -n 0.4 --net mae --adapter vpt_10
python train_rein_other_vfm_ours_three_head.py -d aptos -g 0 -s maeb_vpt_ours_threehead_0.4_10 -n 0.4 --net mae --adapter vpt

python train_rein_other_vfm.py -d ham10000 -g 0 -s dinov2b_vpt_0.4_10 -n 0.4 --net dinov2 --adapter vpt
python train_rein_other_vfm_ours_three_head.py -d ham10000 -g 0 -s dinov2b_vpt_ours_threehead_0.4_10 -n 0.4 --net dinov2 --adapter vpt

python train_rein_other_vfm.py -d ham10000 -g 0 -s dinov1b_vpt_0.4 -n 0.4 --net dinov1 --adapter vpt
python train_rein_other_vfm_ours_three_head.py -d ham10000 -g 0 -s dinov1b_vpt_ours_threehead_0.4_10 -n 0.4 --net dinov1 --adapter vpt

python train_rein_other_vfm.py -d ham10000 -g 0 -s maeb_vpt_0.4 -n 0.4 --net mae --adapter vpt_10
python train_rein_other_vfm_ours_three_head.py -d ham10000 -g 0 -s maeb_vpt_ours_threehead_0.4_10 -n 0.4 --net mae --adapter vpt