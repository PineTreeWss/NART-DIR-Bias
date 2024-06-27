#Author: SongSheng Wang
#Date: 2022-08-18
#Inference the checkpoints on the target folds on validation set, calculate the corresponding validation set BLEU.
#1. Analyze the validation set BLEU result and draw the line chart
#2. Report the best five checkpoints.

import sys
import os
from matplotlib import pyplot as plt
def test():
    result = AnalyzeResult([
        [0,3.1],
        [1,4.1],
        [2,5.1],
        [3,6],
        [4,2],
        [5,5],
        [6,3]
    ],'test_figure')
    with open('CMLMC-KD.log','w') as f:
        f.write('best five epoch:'+str(result))
        f.close()
def main():
    modelfolder = '../../checkpoints/wmt16-de-en-bi-skd-P0-Upp8'
    datafolder = "../../data-bin/wmt16-de-en-original-skd-preprocessed"
    model_file_name_list = getcheckpointfilename(modelfolder)
    checkpoint_num = 136
    BLEU_List = []
    for i in range(checkpoint_num):
        checkpoint_id = i+1
        if checkpoint_id == 76:
            continue
        if checkpoint_id == checkpoint_num:
            checkpoint_id = '_last'
        else:
            checkpoint_id = str(i+1)
        BLEU1 = inferencecheckpoint(checkpoint_id,datafolder,modelfolder,'valid-f',0)
        BLEU2 = inferencebackcheckpoint(checkpoint_id,datafolder,modelfolder,'valid-b',0)
        BLEU_List.append([checkpoint_id,(BLEU1+BLEU2)/2])
    best_five_epoch = AnalyzeResult(BLEU_List,'ro-en-bi.jpg')
    with open('ro-en-valid-Bi.log','w') as f:
        f.write('best five epoch:'+str(best_five_epoch))
        f.close()
def AnalyzeResult(BLEU_List,figure_name):
    sorted_List = sorted(BLEU_List,reverse=1,key=getkey)
    epoch_num_set = []
    valid_loss_set = []
    for item in BLEU_List:
        epoch_num_set.append(item[0])
        valid_loss_set.append(item[1])
    best_epoch = sorted_List[0][0]
    best_bleu = sorted_List[0][1]
    best_five_items = sorted_List[:5]
    best_five_epoch = []
    for item in best_five_items:
        best_five_epoch.append(item[0])
    '''plt.ylabel('Valid Bleu')
    plt.xlabel('epoch')
    plt.title('NAT Training Procedure')
    plt.plot(epoch_num_set, valid_loss_set)
    plt.scatter(best_epoch, best_bleu, marker='o', color='red', s=10, label='First')
    plt.savefig(figure_name)'''
    return best_five_epoch
def getkey(item):
    return item[1]
def getcheckpointfilename(modelfolder):
    #get the file name list in the checkpointfile
    #raise error if the file doesn't end with .pt
    all_file = []
    for f in os.listdir(modelfolder):
        if (f[-1] != 't') | (f[-2] !='p'):
            print('error: checkpoint folder has file ends without .pt')
            exit()
        f_name = os.path.join(modelfolder,f)
        all_file.append(f_name)
    return all_file
def inferencecheckpoint(checkpoint_num,datafolder,modelfolder,resultfolder,device_id=0):
    print('inferencing checkpoint No. '+str(checkpoint_num))
    inference_order = ' CUDA_VISIBLE_DEVICES=0 fairseq-generate '+datafolder+ \
                      ' --gen-subset valid --user-dir fs_plugins --task translation_lev_modified ' \
                      ' --path '+modelfolder+'/checkpoint_'+str(checkpoint_num)+'.pt' \
                      ' --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 --beam 1 ' \
                      ' --remove-bpe --max-tokens 2048 --seed 0 --skip-invalid-size-inputs-valid-test' \
                      ' --model-overrides "{\\"decode_strategy\\":\\"lookahead\\",\\"decode_beta\\":1, \\"decode_direction\\":\\"forward\\"}" > '+str(resultfolder)+'/valid_'+str(checkpoint_num)+'.out'
    os.system(inference_order)
    with open(resultfolder+'/valid_'+str(checkpoint_num)+'.out') as f:
        lines = f.read().splitlines()
        lastline = lines[-1].replace(',', '').split()
        validbleu = float(lastline[6])
        return validbleu
    print('Error: result file not open')
    exit()
def inferencebackcheckpoint(checkpoint_num,datafolder,modelfolder,resultfolder,device_id=0):
    print('inferencing checkpoint No. '+str(checkpoint_num))
    inference_order = ' CUDA_VISIBLE_DEVICES=0 fairseq-generate '+datafolder+ \
                      ' --gen-subset valid --user-dir fs_plugins --task translation_lev_modified ' \
                      ' --path '+modelfolder+'/checkpoint_'+str(checkpoint_num)+'.pt' \
                      ' --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 --beam 1 --skip-invalid-size-inputs-valid-test' \
                      ' --remove-bpe --max-tokens 2048 --seed 0 ' \
                      ' --model-overrides "{\\"decode_strategy\\":\\"lookahead\\",\\"decode_beta\\":1, \\"decode_direction\\":\\"backward\\"}" > '+str(resultfolder)+'/valid_'+str(checkpoint_num)+'.out'
    os.system(inference_order)
    with open(resultfolder+'/valid_'+str(checkpoint_num)+'.out') as f:
        lines = f.read().splitlines()
        lastline = lines[-1].replace(',', '').split()
        validbleu = float(lastline[6])
        return validbleu
    print('Error: result file not open')
    exit()
main()