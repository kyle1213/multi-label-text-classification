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
