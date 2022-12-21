import os

if __name__ == '__main__':
    path = '/SSDe/yyg/RR/Cifar100/student_hl_norm'
    # path = '/SSDd/yyg/Birds/resnet50_baseline'

    method = 'energy'
    # method = 'msp'
    with open('./temp.txt', 'w') as g:
        file = os.path.join(path, '{}_result.txt'.format(method))
        
        with open(file, 'r') as f:
            performances=[]
            lines = f.readlines()
            print(lines[0])
            n=0
            for i in range(len(lines)):
                splits = lines[i].split('\t')
                if splits[0].startswith(method):
                    n += 1
                    # print(lines[i+1].split('\t')[-1][:-2],'\t', lines[i+2].split('\t')[-1][:-2], '\t', lines[i+3].split('\t')[-1][:-2])
                    g.write(lines[i+1].split('\t')[-1][:-1]+'\t'+lines[i+2].split('\t')[-1][:-1]+'\t'+lines[i+3].split('\t')[-1][:-1]+'\t')
                    # if n == 5:
            g.write('{:.2f}\t'.format(float(lines[0].split(' ')[-1][:-1]) *100 ))                            
            g.write('\n')
                    
                
                
        