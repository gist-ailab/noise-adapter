gpu=0
data=ham10000
noiserate=0.4

echo ${gpu} ${data} ${noiserate}
python train_rein_other_vfm.py -d aptos -g 0 -n 0.4 -s bioCLIP_vpt_0.4 --net bioCLIP --adapter vpt
python train_rein_other_vfm.py -d aptos -g 0 -n 0.4 -s bioCLIP_adaptformer_0.4 --net bioCLIP --adapter adaptformer

python train_rein_other_vfm.py -d ham10000 -g 0 -n 0.4 -s bioCLIP_vpt_0.4 --net bioCLIP --adapter vpt
python train_rein_other_vfm.py -d ham10000 -g 0 -n 0.4 -s bioCLIP_adaptformer_0.4 --net bioCLIP --adapter adaptformer