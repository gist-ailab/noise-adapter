import csv

if __name__ == '__main__':
    save_path = 'resnet18_ema_only_0.5'

    with open('/SSDe/yyg/RR/Cifar10/{}/record.txt'.format(save_path)) as f:
        with open('/SSDe/yyg/RR/Cifar10/{}/record.csv'.format(save_path), 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            lines = f.readlines()
            for line in lines:
                print(line[:-1].split('\t'))
                csvwriter.writerow(line[:-1].split('\t'))