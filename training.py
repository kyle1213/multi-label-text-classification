import numpy as np
import torch
from torch import nn

from tqdm import tqdm_notebook

from transformers import AutoModel

from utils import calc_accuracy


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, length):
        self.data = data
        self.tokenized = [tokenizer(d[0], padding='max_length', truncation=True, max_length=length) for d in data]

    def __getitem__(self, idx):
        item = {key: torch.tensor(val) for key, val in self.tokenized[idx].items()}
        item['labels'] = torch.tensor(self.data[idx][1])

        return item

    def __len__(self):
        return len(self.data)


class BBClassifier(nn.Module):
    def __init__(self, model_path, config, num_classes=33, dr_rate=None):
        super(BBClassifier, self).__init__()
        if model_path == 'monologg/kobigbird-bert-base':
            config.attention_type = "block_sparse"
        self.base_model = AutoModel.from_pretrained(model_path, config=config)

        self.classifier = nn.Linear(768, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
            self.dr_rate = dr_rate

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        model_out = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        out = model_out[1] # 6, 1024, 768

        if self.dr_rate:
            out = self.dropout(out)

        return self.classifier(out)


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum(1)


class EarlyStopping():
    def __init__(self, path, patience=5, verbose=False, delta=0):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False # 조기 종료를 의미하며 초기값은 False로 설정
        self.delta = delta # 오차가 개선되고 있다고 판단하기 위한 최소 변화량
        self.path = path
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        # 에포크 만큼 한습이 반복되면서 best_loss가 갱신되고, bset_loss에 진전이 없으면 조기종료 후 모델을 저장
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)

        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'Early Stopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}. Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def make_fold(train_idx, val_idx, train_data, batch_size, tokenizer, length):
    kfold_train = [train_data[i] for i in train_idx]
    kfold_val = [train_data[i] for i in val_idx]

    kfold_train_Dataset = CustomDataset(kfold_train, tokenizer, length)
    kfold_val_Dataset = CustomDataset(kfold_val, tokenizer, length)

    kfold_train_dataloader = torch.utils.data.DataLoader(kfold_train_Dataset, batch_size=batch_size, shuffle=True)
    kfold_val_dataloader = torch.utils.data.DataLoader(kfold_val_Dataset, batch_size=batch_size, shuffle=False) # , collate_fn=lambda x: x

    return kfold_train_dataloader, kfold_val_dataloader


def training(model, kfold_train_dataloader, optimizer, device, loss_fn, scheduler, fold, max_grad_norm, e):
    model.train()
    train_losses = 0
    train_correct = 0
    counter = 0
    _ = 0
    for batch_id, data in enumerate(tqdm_notebook(kfold_train_dataloader)):
        optimizer.zero_grad()
        input_ids = data['input_ids'].long().to(device)
        attention_mask = data['attention_mask'].long().to(device)
        label = data['labels'].long().to(device)
        counter += label.size(0)
        out = model(input_ids, attention_mask)
        train_loss = loss_fn(out, label.float()).mean()
        train_losses += train_loss.cpu().detach()
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        _, tmp_correct = calc_accuracy(out, label.float())
        train_correct += tmp_correct
        scheduler.step()

    print("{}fold epoch {} train acc {} train loss {}".format(fold, e+1, train_correct / counter, train_losses/len(kfold_train_dataloader)))
    print(train_correct, counter)

    return train_losses/len(kfold_train_dataloader), train_correct/counter, scheduler


def validate(model, kfold_val_dataloader, device, loss_fn, fold, e):
    model.eval()
    val_losses = 0
    val_correct = 0
    counter = 0
    _ = 0
    # torch.save(model.state_dict(), './' + str(fold) + 'th_model' + str(e) + '.pt')
    with torch.no_grad():
        for batch_id, data in enumerate(tqdm_notebook(kfold_val_dataloader)):
            input_ids = data['input_ids'].long().to(device)
            attention_mask = data['attention_mask'].long().to(device)
            label = data['labels'].long().to(device)
            counter += label.size(0)
            out = model(input_ids, attention_mask)
            val_loss = loss_fn(out, label.float()).mean()
            val_losses += val_loss.cpu().detach()
            _, tmp_correct = calc_accuracy(out, label)
            val_correct += tmp_correct
        print("{}fold epoch {} val acc {} val losses {}".format(fold, e+1, val_correct/counter, val_losses/len(kfold_val_dataloader)))

    return val_losses/len(kfold_val_dataloader), val_correct/counter