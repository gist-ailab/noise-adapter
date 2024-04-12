gpu=${1}
data=${2}
noiserate=${3}

echo ${gpu} ${data} ${noiserate}
python train_linear.py -g ${gpu} -d ${data} -s dinov2s_linear_${noiserate} -n ${noiserate}
python train_fully.py -g ${gpu} -d ${data} -s dinov2s_full_${noiserate} -n ${noiserate}
python train_rein.py -g ${gpu} -d ${data} -s dinov2s_rein_${noiserate} -n ${noiserate}
