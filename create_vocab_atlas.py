import csv
import re
import os

import time
begin=time.time()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-path', action='store', default=None, dest='path')
parser.add_argument('-t_name', action='store')
parser.add_argument('-v_name', action='store')
args = parser.parse_args()

words={}
flag=3
data=[]
filename=args.path
f=open(filename,'r',encoding='utf-8')
data=f.read().split('\n')
f.close()

strr_store=[]
count=0
ffflag=False
for i in data:
    count=0
    if len(i)==0:
        strr_store.append("\n")
        continue
    i=i.split(',')[1:]
    tmp_data=[]
    for j in range(len(i)):
        countt=True
        candidate=i[j].replace(' ','')
        if len(candidate)==0:
            count=count+1
        else:
            if count!=0:
                tmp_data.append(str(count))
                count=0
                ffflag=True
            #tmp_data.append("PlaceHolder")
        candidate=re.split('\\\|->|_|-|\.|/|:|\(|\)',candidate)#add field wise ones, n/a in atlas datasets
        for kk in candidate:
            if kk==None or len(kk)==0:
                continue
            if countt:
                tmp_data.append(kk)
                countt=False
            else:
                if ffflag:
                    tmp_data.append(kk)
                    ffflag=False
                else:
                    tmp_data.append(kk)
    if count!=0:
        tmp_data.append(str(count))
        count=0
    strr_store.append(" ".join(tmp_data))
    for tmp_key in tmp_data:
        if tmp_key not in words.keys():
            words[tmp_key]=0
        words[tmp_key]=words[tmp_key]+1
word_set=set()
print(time.time()-begin)
min_number=8
for i in words.keys():
    if words[i]>=min_number:
        word_set.add(i)
for i in ["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]", "sim", 'sim_no', 'unlabeled','placeholder']:
    word_set.add(i)

dir_name = os.path.dirname(filename)
t_path = dir_name + '/' + args.t_name
v_path = dir_name + '/' + args.v_name

f1=open(v_path,'w',encoding='utf-8')
f1.write('\n'.join(list(word_set)))
f1.close()

f1=open(t_path,'w',encoding='utf-8')
f1.write("\n".join(strr_store))
f1.close()

