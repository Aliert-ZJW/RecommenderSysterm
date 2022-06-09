import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def processing():
    # 获取数据
    train_data = pd.read_csv('./data/train.csv')
    test_data = pd.read_csv('./data/test.csv')
    lable = train_data['Label']
    del train_data['Label']
    all_data = pd.concat((train_data, test_data))
    del all_data['Id']
    sparse_f = [col for col in all_data.columns if col[0]=='C']
    dense_f = [col for col in all_data.columns if col[0]=='I']
    all_data[sparse_f] = all_data[sparse_f].fillna('-1')
    all_data[dense_f] = all_data[dense_f].fillna(0)
    for f in sparse_f:
        le = LabelEncoder()
        all_data[f] = le.fit_transform(all_data[f])
    mms = MinMaxScaler()
    all_data[dense_f] = mms.fit_transform(all_data[dense_f])
    #将数据处理完毕之后，再进行训练集和测试集的区分
    train = all_data[:train_data.shape[0]]
    test = all_data[train_data.shape[0]:]
    train['Label'] = lable
    train_set, val_set = train_test_split(train, test_size=0.2, random_state=0)
    train_set.reset_index(drop=True, inplace=True)
    val_set.reset_index(drop=True, inplace=True)

    train_set.to_csv('/data/zjw/TJ/DeepCross/data/train_set.csv', index=0)
    val_set.to_csv('/data/zjw/TJ/DeepCross/data/val_set.csv', index=0)
    test.to_csv('/data/zjw/TJ/DeepCross/data/test_set.csv', index=0)
    print('finish')


processing()





