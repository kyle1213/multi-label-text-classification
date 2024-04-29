import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter
from sklearn.model_selection import train_test_split


def filter_null_contents(df):
    df_filtered = df[df['contents'] != '']

    return df_filtered


def create_multi_label1_vector(data):
    pos_vector = np.zeros(11)
    if '브랜드' in data['긍정']:
        pos_vector[0] = 1
    if '디자인' in data['긍정']:
        pos_vector[1] = 1
    if '발볼' in data['긍정내용']:
        pos_vector[2] = 1
    if '발등' in data['긍정내용']:
        pos_vector[3] = 1
    if '안정' in data['긍정내용']:
        pos_vector[4] = 1
    if '접지' in data['긍정내용']:
        pos_vector[4] = 1
    if '쿠션' in data['긍정내용']:
        pos_vector[5] = 1
    if '무게' in data['긍정내용']:
        pos_vector[6] = 1
    if '이물' in data['긍정내용']:
        pos_vector[7] = 1
    if '피로' in data['긍정내용']:
        pos_vector[7] = 1
    if '소재' in data['긍정내용']:
        pos_vector[8] = 1
    if '가격' in data['긍정']:
        pos_vector[9] = 1
    if '품질' in data['긍정']:
        pos_vector[10] = 1

    neg_vector = np.zeros(11)
    if '브랜드' in data['부정']:
        neg_vector[0] = 1
    if '디자인' in data['부정']:
        neg_vector[1] = 1
    if '발볼' in data['부정내용']:
        neg_vector[2] = 1
    if '발등' in data['부정내용']:
        neg_vector[3] = 1
    if '안정' in data['부정내용']:
        neg_vector[4] = 1
    if '접지' in data['부정내용']:
        neg_vector[4] = 1
    if '쿠션' in data['부정내용']:
        neg_vector[5] = 1
    if '무게' in data['부정내용']:
        neg_vector[6] = 1
    if '이물' in data['부정내용']:
        neg_vector[7] = 1
    if '피로' in data['부정내용']:
        neg_vector[7] = 1
    if '소재' in data['부정내용']:
        neg_vector[8] = 1
    if '가격' in data['부정']:
        neg_vector[9] = 1
    if '품질' in data['부정']:
        neg_vector[10] = 1

    ques_vector = np.zeros(11)
    if '브랜드' in data['문의']:
        ques_vector[0] = 1
    if '디자인' in data['문의']:
        ques_vector[1] = 1
    if '발볼' in data['문의내용']:
        ques_vector[2] = 1
    if '발등' in data['문의내용']:
        ques_vector[3] = 1
    if '안정' in data['문의내용']:
        ques_vector[4] = 1
    if '접지' in data['문의내용']:
        ques_vector[4] = 1
    if '쿠션' in data['문의내용']:
        ques_vector[5] = 1
    if '무게' in data['문의내용']:
        ques_vector[6] = 1
    if '이물' in data['문의내용']:
        ques_vector[7] = 1
    if '피로' in data['문의내용']:
        ques_vector[7] = 1
    if '소재' in data['문의내용']:
        ques_vector[8] = 1
    if '가격' in data['문의']:
        ques_vector[9] = 1
    if '품질' in data['문의']:
        ques_vector[10] = 1

    # 멀티 레이블 원핫 벡터 생성
    multi_label_vector = np.concatenate((pos_vector, neg_vector, ques_vector)).tolist()

    return multi_label_vector


def create_multi_label2_vector(data):
    pos_vector = np.zeros(11)
    if '브랜드' in data['긍정']:
        pos_vector[0] = 1
    if '디자인' in data['긍정']:
        pos_vector[1] = 1
    if '발볼' in data['긍정']:
        pos_vector[2] = 1
    if '발등' in data['긍정']:
        pos_vector[3] = 1
    if '안정' in data['긍정']:
        pos_vector[4] = 1
    if '접지' in data['긍정']:
        pos_vector[4] = 1
    if '쿠션' in data['긍정']:
        pos_vector[5] = 1
    if '무게' in data['긍정']:
        pos_vector[6] = 1
    if '이물' in data['긍정']:
        pos_vector[7] = 1
    if '피로' in data['긍정']:
        pos_vector[7] = 1
    if '소재' in data['긍정']:
        pos_vector[8] = 1
    if '가격' in data['긍정']:
        pos_vector[9] = 1
    if '품질' in data['긍정']:
        pos_vector[10] = 1

    neg_vector = np.zeros(11)
    if '브랜드' in data['부정']:
        neg_vector[0] = 1
    if '디자인' in data['부정']:
        neg_vector[1] = 1
    if '발볼' in data['부정']:
        neg_vector[2] = 1
    if '발등' in data['부정']:
        neg_vector[3] = 1
    if '안정' in data['부정']:
        neg_vector[4] = 1
    if '접지' in data['부정']:
        neg_vector[4] = 1
    if '쿠션' in data['부정']:
        neg_vector[5] = 1
    if '무게' in data['부정']:
        neg_vector[6] = 1
    if '이물' in data['부정']:
        neg_vector[7] = 1
    if '피로' in data['부정']:
        neg_vector[7] = 1
    if '소재' in data['부정']:
        neg_vector[8] = 1
    if '가격' in data['부정']:
        neg_vector[9] = 1
    if '품질' in data['부정']:
        neg_vector[10] = 1

    ques_vector = np.zeros(11)
    if '브랜드' in data['문의']:
        ques_vector[0] = 1
    if '디자인' in data['문의']:
        ques_vector[1] = 1
    if '발볼' in data['문의']:
        ques_vector[2] = 1
    if '발등' in data['문의']:
        ques_vector[3] = 1
    if '안정' in data['문의']:
        ques_vector[4] = 1
    if '접지' in data['문의']:
        ques_vector[4] = 1
    if '쿠션' in data['문의']:
        ques_vector[5] = 1
    if '무게' in data['문의']:
        ques_vector[6] = 1
    if '이물' in data['문의']:
        ques_vector[7] = 1
    if '피로' in data['문의']:
        ques_vector[7] = 1
    if '소재' in data['문의']:
        ques_vector[8] = 1
    if '가격' in data['문의']:
        ques_vector[9] = 1
    if '품질' in data['문의']:
        ques_vector[10] = 1

    # 멀티 레이블 원핫 벡터 생성
    multi_label_vector = np.concatenate((pos_vector, neg_vector, ques_vector)).tolist()

    return multi_label_vector


def df_to_list1(df_filled, labels):
    data_list = []

    for seq, title, contents, label in zip(df_filled['seq'], df_filled['title'], df_filled['contents'], labels):
        data = [seq]
        data.append(title + ' ' + str(contents))
        data.append(label)
        data_list.append(data)

    return data_list


def df_to_list2(df_filled, labels):
    data_list = []

    for url, contents, label in zip(df_filled['link'], df_filled['contents'], labels):
        data = [url]
        data.append(str(contents))
        data.append(label)
        data_list.append(data)

    return data_list


def sum_list_values(lst):
    return [sum(x) for x in zip(*lst)]


def make_data_list(df_combined):
    data_list = []

    for x, y in zip(df_combined['x'], df_combined['y']):
        data = [x]
        new_y = [1 if _ >= 1 else 0 for _ in y]
        data.append(new_y)
        data_list.append(data)

    return data_list


def data_preprocess(data_list):
    x_data_list = [x[0] for x in data_list]
    y_data_list = [y[1] for y in data_list]

    x_preprocessed_data_list = list(map(lambda x: re.sub("[^ A-Za-z0-9가-힣]", "", x), x_data_list))
    x_preprocessed_data_list = list(map(lambda x: re.sub("[ +]", " ", x), x_preprocessed_data_list))

    preprocessed_data_list = [[x, y] for x, y in zip(x_preprocessed_data_list, y_data_list)]

    return preprocessed_data_list


def calculate_IRLbl(labels):
    label_counts_1 = np.sum(np.array(labels) == 1, axis=0)
    for i, c in enumerate(label_counts_1):
        print('class ', str(i), "'s imbalance ratio is: ", max(label_counts_1)/max(c, 1))


def calculate_mean_IRLbl(labels):
    mean = 0
    label_counts_1 = np.sum(np.array(labels) == 1, axis=0)
    for i, c in enumerate(label_counts_1):
      mean += max(label_counts_1)/max(c, 1)
    print('mean IRlbl : ', mean/len(label_counts_1))


def histogram_0and1(labels):
    label_counts_0 = np.sum(np.array(labels) == 0, axis=0)
    label_counts_1 = np.sum(np.array(labels) == 1, axis=0)

    # 히스토그램 플롯
    bar_width = 0.35
    index = np.arange(len(label_counts_0))
    plt.bar(index, label_counts_0, bar_width, label='0')
    plt.bar(index + bar_width, label_counts_1, bar_width, label='1')
    plt.xlabel('Class Index')
    plt.ylabel('Frequency')
    plt.title('Histogram of Multi-label Classes (0s and 1s)')
    plt.xticks(index + bar_width / 2, range(len(label_counts_0)))
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.show()


def histogram_1s(labels):
    label_counts_1 = np.sum(np.array(labels) == 1, axis=0)
    # 각 인덱스에서 1의 frequency 기록
    index_freq = {}
    for i, freq in enumerate(label_counts_1):
        if freq > 0:
            index_freq[i] = freq

    # 히스토그램 플롯
    plt.bar(index_freq.keys(), index_freq.values())
    plt.xlabel('Class Index')
    plt.ylabel('Frequency of 1s')
    plt.title('Frequency of 1s in Multi-label Classes')
    plt.xticks(range(len(labels[0])))
    plt.grid(axis='y', alpha=0.75)

    # 각 인덱스별 1의 frequency 출력
    for i, freq in index_freq.items():
        plt.text(i, freq, str(freq), ha='center', va='bottom')

    plt.show()


def histogram_label_set(labels):
    # 라벨의 등장 횟수를 카운트
    label_counts = Counter(tuple(label) for label in labels)

    # 히스토그램 플롯
    labels_unique = list(label_counts.keys())
    labels_count = list(label_counts.values())

    print('unique label set: ', len(labels_unique))
    print('label coounts: ', labels_count)
    print('label unique: ', labels_unique)

    plt.bar(range(len(labels_unique)), labels_count)
    plt.xlabel('Label')
    plt.ylabel('Frequency')
    plt.title('Histogram of Label Frequencies')
    plt.xticks(range(len(labels_unique)), [str(label) for label in labels_unique])
    plt.grid(axis='y', alpha=0.75)
    plt.show()


def data_prep():
    df1 = pd.read_excel("./data/러닝화_분류용 데이터정리.xlsx")
    df2 = pd.read_excel("./data/트위터_러닝화.xlsx")

    df1_filtered = filter_null_contents(df1)
    df2_filtered = filter_null_contents(df2)

    df1_filled = df1_filtered.fillna('-')
    df2_filled = df2_filtered.fillna('-')

    labels1 = []
    for index, data in df1_filled.iterrows():
        labels1.append(create_multi_label1_vector(data))

    labels2 = []
    for index, data in df2_filled.iterrows():
        labels2.append(create_multi_label2_vector(data))

    tmp_data_list1 = df_to_list1(df1_filled, labels1)
    tmp_data_list2 = df_to_list2(df2_filled, labels2)

    new_df1 = pd.DataFrame(tmp_data_list1, columns=['seq', 'x', 'y'])
    new_df2 = pd.DataFrame(tmp_data_list2, columns=['url', 'x', 'y'])

    df1_y_combined = new_df1.groupby('seq')['y'].apply(sum_list_values).reset_index()
    df2_y_combined = new_df2.groupby('url')['y'].apply(sum_list_values).reset_index()

    df1_x_unique = new_df1.drop_duplicates('seq')[['seq', 'x']]
    df2_x_unique = new_df2.drop_duplicates('url')[['url', 'x']]

    df1_combined = pd.merge(df1_x_unique, df1_y_combined, on='seq')
    df2_combined = pd.merge(df2_x_unique, df2_y_combined, on='url')

    data_list1 = make_data_list(df1_combined)
    data_list2 = make_data_list(df2_combined)

    preprocessed_data_list1 = data_preprocess(data_list1)
    preprocessed_data_list2 = data_preprocess(data_list2)

    train_data, test_data = train_test_split(preprocessed_data_list1, test_size=0.3, shuffle=False, random_state=0)
    train_data = train_data + preprocessed_data_list2

    return train_data, test_data


def data_eda(data):
    data_y = [d[1] for d in data]

    calculate_IRLbl(data_y)
    calculate_mean_IRLbl(data_y)

    histogram_0and1(data_y)
    histogram_1s(data_y)
    histogram_label_set(data_y)


if __name__ == '__main__':
    train_data, test_data = data_prep()

    data_eda(train_data)
    data_eda(test_data)