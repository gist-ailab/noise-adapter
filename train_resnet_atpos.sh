for i in 0.0 0.2;
    do python train_resnet_ours_three_head.py -d ham10000 -g 1 -s resnet50_reins_3head_$i -n $i
done

for i in 0.0 0.2 0.4;
    do python train_resnet_ours_three_head.py -d aptos -g 1 -s resnet50_reins_3head_$i -n $i
done


# do python train_linear.py -g ${gpu} -d ${data} -s dinov2s_linear_${noiserate} -n ${noiserate}
# python train_fully.py -g ${gpu} -d ${data} -s dinov2s_full_${noiserate} -n ${noiserate}
# python train_resnet.py -g ${gpu} -d ${data} -s dinov2s_rein_${noiserate} -n ${noiserate}
