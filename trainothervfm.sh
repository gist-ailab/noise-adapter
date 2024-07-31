gpu=0
data=ham10000
noiserate=0.4

echo ${gpu} ${data} ${noiserate}
python train_linear_other_vfm.py -d aptos -g 0 -s dinov1b_linear_0.4 -n 0.4 --net dinov1
python train_rein_other_vfm.py -d aptos -g 0 -s dinov1b_rein_0.4 -n 0.4 --net dinov1
python train_rein_other_vfm_ours_three_head.py -d aptos -g 0 -s dinov1b_rein_ours_threehead_0.4 -n 0.4 --net dinov1

python train_linear_other_vfm.py -d aptos -g 0 -s maeb_linear_0.4 -n 0.4 --net mae
python train_rein_other_vfm.py -d aptos -g 0 -s maeb_rein_0.4 -n 0.4 --net mae
python train_rein_other_vfm_ours_three_head.py -d aptos -g 0 -s maeb_rein_ours_threehead_0.4 -n 0.4 --net mae

python train_linear_other_vfm.py -d aptos -g 0 -s dinov2b_linear_0.4 -n 0.4 --net dinov2
python train_rein_other_vfm.py -d aptos -g 0 -s dinov2b_rein_0.4 -n 0.4 --net dinov2
python train_rein_other_vfm_ours_three_head.py -d aptos -g 0 -s dinov2b_rein_ours_threehead_0.4 -n 0.4 --net dinov2