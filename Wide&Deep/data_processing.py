import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def sparsFeature(feat, feat_num, embed_dim=4):
    """
    create dictionary for sparse feature
    :param feat: feature_name
    :param feat_num: the total number of sparse features that do not repeat
    :param embed_dim: embedding dimension
    :return
    """
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}

def denseFeature(feat):
    """
    create dictionary for dense feature
    :param feat: dense feature name
    : return
    """
    return {'feat': feat}

def processing(embeeding_dim=8):
    # 获取数据
    train_data = pd.read_csv('Wide&Deep/data/train.csv')
    test_data = pd.read_csv('TJ/Wide&Deep/data/test.csv')
    lable = train_data['Label']
    del train_data['Label']
    all_data = pd.concat((train_data, test_data))
    del all_data['Id']
    sparse_f = [col for col in all_data.columns if col[0]=='C']
    dense_f = [col for col in all_data.columns if col[0]=='I']
    all_data[sparse_f] = all_data[sparse_f].fillna('-1')
    all_data[dense_f] = all_data[dense_f].fillna(0)

    feature_columns = [[denseFeature(feat) for feat in dense_f] +
                       [sparsFeature(feat, len(all_data[feat].unique()), embeeding_dim) for feat in sparse_f]]
    print(feature_columns)
    np.save('data/feat__col.npy', feature_columns)

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

    train_set.to_csv('Wide&Deep/process_data/train_set.csv', index=0)
    val_set.to_csv('Wide&Deep/process_data/val_set.csv', index=0)
    test.to_csv('Wide&Deep/process_data/test_set.csv', index=0)
    print('finish')


processing()





