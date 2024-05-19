gpu=1
data=organcmnist
noiserate=0.6

echo ${gpu} ${data} ${noiserate}

python train_linear.py -g ${gpu} -d ${data} -s dinov2s_linear_${noiserate} -n ${noiserate}
python train_fully.py -g ${gpu} -d ${data} -s dinov2s_full_${noiserate} -n ${noiserate}
python train_rein.py -g ${gpu} -d ${data} -s dinov2s_rein_${noiserate} -n ${noiserate}
python train_rein_jocor.py -g ${gpu} -d ${data} -s dinov2s_rein_jocor_${noiserate} -n ${noiserate}

python train_rein_ours.py -g ${gpu} -d ${data} -s dinov2s_rein_ours_0.0 -n 0.0
python train_rein_ours.py -g ${gpu} -d ${data} -s dinov2s_rein_ours_0.6 -n 0.6



