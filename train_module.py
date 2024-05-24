import numpy as np
import torch
import torch.utils.data
from torch import nn
from transformers import AutoModel, AutoTokenizer
from utils import calc_accuracy
from data import FullEasyDataAugmentation, PartialEasyDataAugmentation, miniEasyDataAugmentation
from tqdm.notebook import tqdm


class Dataset(torch.utils.data.Dataset):
    def __init__(self, model_path, data, tokenizer, length):
        self.model_path = model_path
        self.data = data
        self.tokenizer = tokenizer
        self.length = length

    def __getitem__(self, idx):
        if self.model_path == 'monologg/kobigbird-bert-base':
            item = {key: torch.tensor(val) for key, val in self.tokenizer(self.data[idx][0], padding='max_length', truncation=True, max_length=self.length).items()}
            item['labels'] = torch.tensor(self.data[idx][1])

            return item

        elif self.model_path == "beomi/KcELECTRA-base-v2022":
            inputs = self.tokenizer(self.data[idx][0], return_tensors='pt', truncation=True,
                                    max_length=512, pad_to_max_length=True, add_special_tokens=True)

            input_ids = inputs['input_ids'][0]
            attention_mask = inputs['attention_mask'][0]
            token_type_ids = inputs['token_type_ids'][0]

            y = torch.tensor(self.data[idx][1])

            return input_ids, attention_mask, token_type_ids, y

    def __len__(self):
        return len(self.data)


class BBClassifier(nn.Module):
    def __init__(self, model_path, config, num_classes, dr_rate=None):
        super(BBClassifier, self).__init__()
        if model_path == 'monologg/kobigbird-bert-base':
            config.attention_type = "block_sparse"
        self.base_model = AutoModel.from_pretrained(model_path, config=config)

        self.classifier = nn.Linear(768, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
            self.dr_rate = dr_rate
        else:
            self.dr_rate = None

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        model_out = self.base_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        out = model_out[1]

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


class EarlyStopping:
    def __init__(self, path, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False  # 조기 종료를 의미하며 초기값은 False로 설정
        self.delta = delta  # 오차가 개선되고 있다고 판단하기 위한 최소 변화량
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


def make_fold(model_path, train_idx, val_idx, train_data, batch_size, tokenizer, length):
    if model_path == "monologg/kobigbird-bert-base":
        kfold_train = [train_data[i] for i in train_idx]
        kfold_val = [train_data[i] for i in val_idx]

        eda_train = FullEasyDataAugmentation(kfold_train)

        kfold_train_Dataset = Dataset(model_path, eda_train, tokenizer, length)
        kfold_val_Dataset = Dataset(model_path, kfold_val, tokenizer, length)

        kfold_train_dataloader = torch.utils.data.DataLoader(kfold_train_Dataset, batch_size=batch_size, shuffle=True)
        kfold_val_dataloader = torch.utils.data.DataLoader(kfold_val_Dataset, batch_size=batch_size, shuffle=False)

        return kfold_train_dataloader, kfold_val_dataloader

    elif model_path == "beomi/KcELECTRA-base-v2022":
        tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base-v2022")

        kfold_train = [train_data[i] for i in train_idx]
        kfold_val = [train_data[i] for i in val_idx]

        eda_train = FullEasyDataAugmentation(kfold_train)

        kfold_train_Dataset = Dataset(model_path, eda_train, tokenizer, length)
        kfold_val_Dataset = Dataset(model_path, kfold_val, tokenizer, length)

        kfold_train_dataloader = torch.utils.data.DataLoader(kfold_train_Dataset, batch_size=batch_size, shuffle=True)
        kfold_val_dataloader = torch.utils.data.DataLoader(kfold_val_Dataset, batch_size=batch_size, shuffle=False)

        return kfold_train_dataloader, kfold_val_dataloader


def train_model(model_path, model, kd_model, do_kd, kfold_train_dataloader, optimizer, device, loss_fn, kd_loss_func, scheduler, fold, max_grad_norm, e):
    model.train()
    train_losses = 0
    train_accuracy = 0
    if model_path == "monologg/kobigbird-bert-base":
        for batch_id, data in enumerate(tqdm(kfold_train_dataloader)):
            optimizer.zero_grad()

            input_ids = data['input_ids'].long().to(device)
            attention_mask = data['attention_mask'].long().to(device)
            token_type_ids = data['token_type_ids'].long().to(device)
            label = data['labels'].long().to(device)

            out = model(input_ids, attention_mask, token_type_ids)
            train_loss = loss_fn(out, label.float()).mean()

            if do_kd:
                with torch.no_grad():
                    kd_out = kd_model(input_ids, attention_mask)
                    kd_loss = kd_loss_func(out, kd_out)
                train_loss += 1 * kd_loss
                tmp_kd_model = model_ensemble(kd_model, model, e * len(kfold_train_dataloader) + batch_id)
                kd_model.load_state_dict(tmp_kd_model)

            train_losses += train_loss.cpu().detach()
            accuracy, _ = calc_accuracy(out, label.float())
            train_accuracy += accuracy

            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule

        print("{}fold epoch {} train acc {} train loss {}".
              format(fold, e + 1, train_accuracy / len(kfold_train_dataloader),
                     train_losses / len(kfold_train_dataloader)))

        return train_losses / len(kfold_train_dataloader), train_accuracy / len(kfold_train_dataloader), scheduler

    else:  # elif model_path == "beomi/KcELECTRA-base-v2022":
        for batch_id, (input_ids_batch, attention_masks_batch, token_type_ids_batch, label) in enumerate(
                kfold_train_dataloader):
            optimizer.zero_grad()

            input_ids_batch = input_ids_batch.long().to(device)
            attention_masks_batch = attention_masks_batch.long().to(device)
            token_type_ids_batch = token_type_ids_batch.long().to(device)
            label = label.long().to(device)

            out = model(input_ids_batch, attention_masks_batch, token_type_ids_batch)[0]
            train_loss = loss_fn(out, label.float()).mean()

            if do_kd:
                with torch.no_grad():
                    kd_out = kd_model(input_ids_batch, attention_masks_batch, token_type_ids_batch)[0]
                kd_loss = kd_loss_func(out, kd_out)
                train_loss += 1 * kd_loss
                tmp_kd_model = model_ensemble(kd_model, model, e * len(kfold_train_dataloader) + batch_id)
                kd_model.load_state_dict(tmp_kd_model)

            train_losses += train_loss.cpu().detach()
            accuracy, _ = calc_accuracy(out, label.float())
            train_accuracy += accuracy

            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule

        print("{}fold epoch {} train acc {} train loss {}".
              format(fold, e + 1, train_accuracy / len(kfold_train_dataloader),
                     train_losses / len(kfold_train_dataloader)))

        return train_losses / len(kfold_train_dataloader), train_accuracy / len(kfold_train_dataloader), scheduler


def validate_model(model_path, model, kfold_val_dataloader, device, loss_fn, fold, e):
    model.eval()
    val_losses = 0
    val_accuracy = 0

    if model_path == "monologg/kobigbird-bert-base":
        with torch.no_grad():
            for batch_id, data in enumerate(kfold_val_dataloader):
                input_ids = data['input_ids'].long().to(device)
                attention_mask = data['attention_mask'].long().to(device)
                label = data['labels'].long().to(device)

                out = model(input_ids, attention_mask)
                val_loss = loss_fn(out, label.float()).mean()
                val_losses += val_loss.cpu().detach()
                accuracy, _ = calc_accuracy(out, label)
                val_accuracy += accuracy

            print("{}fold epoch {} val acc {} val losses {}".format(fold, e+1, val_accuracy/len(kfold_val_dataloader), val_losses/len(kfold_val_dataloader)))

        return val_losses/len(kfold_val_dataloader), val_accuracy/len(kfold_val_dataloader)

    else:  # elif model_path == "beomi/KcELECTRA-base-v2022":
        with torch.no_grad():
            for batch_id, (input_ids_batch, attention_masks_batch, token_type_ids_batch, y_batch) in enumerate(
                    kfold_val_dataloader):
                input_ids_batch = input_ids_batch.long().to(device)
                attention_masks_batch = attention_masks_batch.long().to(device)
                token_type_ids_batch = token_type_ids_batch.long().to(device)
                y_batch = y_batch.long().to(device)

                out = model(input_ids_batch, attention_masks_batch, token_type_ids_batch)[0]
                val_loss = loss_fn(out, y_batch.float()).mean()
                val_losses += val_loss.cpu().detach()
                _, tmp_correct = calc_accuracy(out, y_batch)
                val_accuracy += tmp_correct
            print(
                "{}fold epoch {} val acc {} val losses {}".format(fold, e + 1, val_accuracy / len(kfold_val_dataloader),
                                                                  val_losses / len(kfold_val_dataloader)))

        return val_losses / len(kfold_val_dataloader), val_accuracy / len(kfold_val_dataloader)


def model_ensemble(prev_model, curr_model, n):
    W0 = prev_model.state_dict()
    W1 = curr_model.state_dict()

    for key in W0:
        W1[key] = W0[key] + (W1[key] - W0[key]) / (n+1)

    return W1
