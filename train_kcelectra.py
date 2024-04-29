from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

from sklearn.model_selection import KFold

from data import *
from utils import *
from training import *


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __getitem__(self, i):
        inputs = self.tokenizer(
        self.dataset[i][0],
        return_tensors='pt',
        truncation=True,
        max_length=512,
        pad_to_max_length=True,
        add_special_tokens=True
        )

        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]
        token_type_ids = inputs['token_type_ids'][0]

        y = torch.tensor(self.dataset[i][1])

        return input_ids, attention_mask, token_type_ids, y

    def __len__(self):
        return (len(self.dataset))


def make_fold(train_idx, val_idx, train_data, batch_size):
    tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base-v2022")

    kfold_train = [train_data[i] for i in train_idx]
    kfold_val = [train_data[i] for i in val_idx]

    kfold_train_Dataset = CustomDataset(kfold_train, tokenizer)
    kfold_val_Dataset = CustomDataset(kfold_val, tokenizer)

    kfold_train_dataloader = torch.utils.data.DataLoader(kfold_train_Dataset, batch_size=batch_size, shuffle=True)
    kfold_val_dataloader = torch.utils.data.DataLoader(kfold_val_Dataset, batch_size=batch_size,
                                                       shuffle=False)  # , collate_fn=lambda x: x

    return kfold_train_dataloader, kfold_val_dataloader


def training(model, kfold_train_dataloader, optimizer, device, loss_fn, scheduler, fold, e):
    model.train()
    train_losses = 0
    train_correct = 0
    counter = 0
    _ = 0
    for batch_id, (input_ids_batch, attention_masks_batch, token_type_ids_batch, y_batch) in enumerate(tqdm_notebook(kfold_train_dataloader)):
        optimizer.zero_grad()
        input_ids_batch = input_ids_batch.long().to(device)
        attention_masks_batch = attention_masks_batch.long().to(device)
        token_type_ids_batch = token_type_ids_batch.long().to(device)
        y_batch = y_batch.long().to(device)
        out = model(input_ids_batch, attention_masks_batch, token_type_ids_batch)[0]
        counter += y_batch.size(0)
        train_loss = loss_fn(out, y_batch.float()).mean()
        train_losses += train_loss.cpu().detach()
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        _, tmp_correct = calc_accuracy(out, y_batch.float())
        train_correct += tmp_correct

        scheduler.step()  # Update learning rate schedule
    print("{}fold epoch {} train acc {} train loss {}".format(fold, e+1, train_correct / counter, train_losses/len(kfold_train_dataloader)))

    return train_losses/len(kfold_train_dataloader), train_correct/counter, scheduler


def validate(model, kfold_val_dataloader, device, loss_fn, fold, e):
    model.eval()
    val_losses = 0
    val_correct = 0
    counter = 0
    _ = 0
    with torch.no_grad():
        for batch_id, (input_ids_batch, attention_masks_batch, token_type_ids_batch, y_batch) in enumerate(tqdm_notebook(kfold_val_dataloader)):
            input_ids_batch = input_ids_batch.long().to(device)
            attention_masks_batch = attention_masks_batch.long().to(device)
            token_type_ids_batch = token_type_ids_batch.long().to(device)
            y_batch = y_batch.long().to(device)
            out = model(input_ids_batch, attention_masks_batch, token_type_ids_batch)[0]
            counter += y_batch.size(0)
            val_loss = loss_fn(out, y_batch.float()).mean()
            val_losses += val_loss.cpu().detach()
            _, tmp_correct = calc_accuracy(out, y_batch)
            val_correct += tmp_correct
        print("{}fold epoch {} val acc {} val losses {}".format(fold, e+1, val_correct/counter, val_losses/len(kfold_val_dataloader)))

    return val_losses/len(kfold_val_dataloader), val_correct/counter


device = connect_cuda()

train_data, test_data = data_prep()

tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base-v2022")

length = 512
max_len = length
batch_size = 16
warmup_ratio = 0.1
num_epochs = 100
max_grad_norm = 1
log_interval = 200
learning_rate = 5e-5

kfold = KFold(n_splits=5, shuffle=True, random_state=0)
kfolds = kfold.split(train_data)

train_losses_list_per_fold = []
train_accs_per_fold = []

val_losses_list_per_fold = []
val_accs_per_fold = []

for fold, (train_idx, val_idx) in enumerate(kfolds):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    model = AutoModelForSequenceClassification.from_pretrained('beomi/KcELECTRA-base-v2022', num_labels=33, problem_type='multi_label_classification').to(device)
    earlystopping = EarlyStopping(path='./model/kcelectra_' + str(fold) + 'fold_model.pt', patience=10, verbose=True, delta=0)

    kfold_train_dataloader, kfold_val_dataloader = make_fold(train_idx, val_idx, train_data, batch_size)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    loss_fn = AsymmetricLoss()

    t_total = len(kfold_train_dataloader) * num_epochs
    warmup_step = int(t_total * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

    for e in range(num_epochs):
        train_loss, train_acc, scheduler = training(model, kfold_train_dataloader, optimizer, device, loss_fn, scheduler, fold, e)
        val_loss, val_acc = validate(model, kfold_val_dataloader, device, loss_fn, fold, e)

        earlystopping(val_loss, model)
        if earlystopping.early_stop:
          break
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

    train_accs_per_fold.append(train_accs)
    val_accs_per_fold.append(val_accs)
    train_losses_list_per_fold.append(train_losses)
    val_losses_list_per_fold.append(val_losses)