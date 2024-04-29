import torch


def connect_cuda():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    return device


def calc_accuracy(xs, ys):
    xs = torch.where(xs <= 0, torch.tensor(0), torch.tensor(1))
    train_acc = 0

    for x, y in zip(xs, ys):
        if all(x == y):
            train_acc += 1

    train_cor = train_acc
    train_acc = train_acc/len(xs)
    return train_acc, train_cor


def acc(predictions, targets):
    total_acc = 0
    for pred, target in zip(predictions, targets):
        if all(pred == target):
            total_acc += 1
    mean_acc = total_acc / len(predictions)
    return mean_acc


def calculate_one_accuracy(predictions, targets):
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


def calculate_zero_accuracy(predictions, targets):
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


def calculate_class_accuracy(predictions, targets, class_num):
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
