import os
import re
from collections import Counter

import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import (train_test_split)
from sklearn.preprocessing import LabelEncoder


# 定义一个函数来读取txt文件并转换为DataFrame
def read_text_file(file_path, sep='\t'):
    data = pd.read_csv(file_path, sep=sep, header=None, names=['column1', 'column2', '...'])
    return data


# 读取原始txt文件
# 假设txt文件中的数据是以制表符分隔的，且没有表头
data = read_text_file("C:/Users/Admin/Desktop/机器学习大作业/cnews.train.txt")

# 随机分割数据集为训练集、验证集和测试集（按比例 8:1:1）
train, temp = train_test_split(data, test_size=0.2, random_state=42)
valid, test = train_test_split(temp, test_size=0.5, random_state=42)


# 将DataFrame保存为txt文件的函数
def save_dataframe_to_txt(dataframe, file_path, sep='\t'):
    dataframe.to_csv(file_path, sep=sep, index=False, header=False)
# 将分割后的数据保存为新的txt文件


save_dataframe_to_txt(train, 'train.txt')
save_dataframe_to_txt(valid, 'valid.txt')
save_dataframe_to_txt(test, 'test.txt')

# 设置数据读取、模型、结果保存路径
base_dir = './'
train_dir = os.path.join(base_dir, 'train.txt')
test_dir = os.path.join(base_dir, 'test.txt')
val_dir = os.path.join(base_dir, 'val.txt')


def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append((content))
                    labels.append(label)
            except:
                pass
    return contents, labels


train_contents, train_labels = read_file(train_dir)
test_contents, test_labels = read_file(test_dir)
val_counts = Counter(train_labels)


#去除文本中的无意义字符（只保留中英文和数字）
def clear_character(sentence):
    pattern1 = '\[.*?\]'
    pattern2 = re.compile('[^\u4e00-\u9fa5^a-z^A-Z^0-9]')
    line1 = re.sub(pattern1, '', sentence)
    line2 = re.sub(pattern2, '', line1)
    new_sentence = ''.join(line2.split())  #去除空白
    return new_sentence


train_text = list(map(lambda s: clear_character(s), train_contents))
test_text = list(map(lambda s: clear_character(s), test_contents))
train_seg_text = list(map(lambda s: jieba.lcut(s), train_text))
test_seg_text = list(map(lambda s: jieba.lcut(s), test_text))


stop_words_path = "0.txt"


def get_stop_words():
    file = open(stop_words_path, 'rb').read().decode('gbk').split('\r\n')
    return set(file)


stopwords = get_stop_words()
# 去掉文本中的停用词


def drop_stopwords(line, stopwords):
    line_clean = []
    for word in line:
        if word in stopwords:
            continue
        line_clean.append(word)
    return line_clean


train_st_text = list(map(lambda s: drop_stopwords(s, stopwords), train_seg_text))
test_st_text = list(map(lambda s: drop_stopwords(s, stopwords), test_seg_text))

le = LabelEncoder()
le.fit(train_labels)
LabelEncoder()

label_train_id = le.transform(train_labels)
label_test_id = le.transform(test_labels)

train_c_text = list(map(lambda s: ' '.join(s), train_st_text))
test_c_text = list(map(lambda s: ' '.join(s), test_st_text))

tfidf_model = TfidfVectorizer(binary=False, token_pattern=r"(?u)\b\w+\b")
train_Data = tfidf_model.fit_transform(train_c_text)
test_Data = tfidf_model.transform(test_c_text)

'''LR模型分类训练'''
classifier = LogisticRegression()
classifier.fit(train_Data, label_train_id)
pred = classifier.predict(test_Data)

print(classification_report(label_test_id, pred, digits=4))
