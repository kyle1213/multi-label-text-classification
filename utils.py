import torch
from sklearn.metrics import classification_report, f1_score


def connect_cuda():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    return device


def micro_calculate_f1_score(targets, predictions):
    f1 = f1_score(targets, predictions, average='micro')

    return f1


def macro_calculate_f1_score(targets, predictions):
    f1 = f1_score(targets, predictions, average='macro')

    return f1


def samples_calculate_f1_score(targets, predictions):
    f1 = f1_score(targets, predictions, average='samples')

    return f1


def weighted_calculate_f1_score(targets, predictions):
    f1 = f1_score(targets, predictions, average='weighted')

    return f1


def calc_accuracy(xs, ys):
    xs = torch.where(xs <= 0, torch.tensor(0), torch.tensor(1))
    train_acc = 0

    for x, y in zip(xs, ys):
        if all(x == y):
            train_acc += 1

    train_cor = train_acc
    train_acc = train_acc/len(xs)
    return train_acc, train_cor


def acc(targets, predictions):
    total_acc = 0
    for pred, target in zip(predictions, targets):
        if all(pred == target):
            total_acc += 1
    mean_acc = total_acc / len(predictions)
    return mean_acc


def partial_acc(targets, predictions, p):
    total_acc = 0
    for pred, target in zip(predictions, targets):
        match_count = 0
        for i in range(len(pred)):
            if pred[i] == target[i]:
                match_count += 1
            if match_count >= len(pred)*p:
                total_acc += 1
                break
    mean_acc = total_acc / len(predictions)
    return mean_acc


def calculate_one_accuracy(targets, predictions):
    correct_count = 0
    total_count = 0
    for pred, target in zip(predictions, targets):
        for p, t in zip(pred, target):
            if p == 1 and t == 1:  # 1로 예측하고 정답이 1인 경우
                correct_count += 1
            if t == 1:
                total_count += 1
    accuracy = correct_count / total_count
    return accuracy


def calculate_zero_accuracy(targets, predictions):
    correct_count = 0
    total_count = 0
    for pred, target in zip(predictions, targets):
        for p, t in zip(pred, target):
            if p == 0 and t == 0:  # 1로 예측하고 정답이 1인 경우
                correct_count += 1
            if t == 0:
                total_count += 1
    accuracy = correct_count / total_count
    return accuracy


def calculate_class_accuracy(targets, predictions, class_num):
    zero_correct_count = 0
    total_zero_count = 0
    one_correct_count = 0
    total_one_count = 0

    for pred, target in zip(predictions, targets):
        if pred[class_num] == 0 and target[class_num] == 0:
            zero_correct_count += 1
        if target[class_num] == 0:
            total_zero_count += 1
        if pred[class_num] == 1 and target[class_num] == 1:
            one_correct_count += 1
        if target[class_num] == 1:
            total_one_count += 1

    if total_one_count != 0:
        one_accuracy = one_correct_count / total_one_count
    else:
        one_accuracy = 999
    if total_zero_count != 0:
        zero_accuracy = zero_correct_count / total_zero_count
    else:
        zero_accuracy = 999
    return zero_accuracy, one_accuracy, total_one_count


def f1_report(targets, predictions):
    return classification_report(targets, predictions)


def metrics(y, y_hat):
    print('micro: ', micro_calculate_f1_score(y, y_hat))
    print('samples: ', samples_calculate_f1_score(y, y_hat))
    print('macro: ', macro_calculate_f1_score(y, y_hat))
    print('weighted: ', weighted_calculate_f1_score(y, y_hat))
    print('acc: ', acc(y, y_hat))
    print('one_acc: ', calculate_one_accuracy(y, y_hat))
    print('zero_acc: ', calculate_zero_accuracy(y, y_hat))
    print('50%-partial acc: ', partial_acc(y, y_hat, 0.5))
    print('80%-partial acc: ', partial_acc(y, y_hat, 0.8))
    print('90%-partial acc: ', partial_acc(y, y_hat, 0.9))
    print('95%-partial acc: ', partial_acc(y, y_hat, 0.95))
    print(print(f1_report(y, y_hat)))