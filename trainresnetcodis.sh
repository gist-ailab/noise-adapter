
for i in 0.0 0.2 0.4;
    do python train_resnet_codis.py -d ham10000 -s resnet50_codis_$i -n $i;
done

for i in 0.0 0.2 0.4;
    do python train_resnet_codis.py -d aptos -s resnet50_codis_$i -n $i;
done
