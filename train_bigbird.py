from transformers import AutoTokenizer, AutoConfig

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

from sklearn.model_selection import KFold

from data import *
from utils import *
from training import AsymmetricLoss, EarlyStopping, BBClassifier, kobigbird_training, kobigbird_Dataset, kobigbird_validate, kobigbird_make_fold


device = connect_cuda()

train_data, test_data = data_prep()
print(len(train_data), len(test_data))

model_path = "monologg/kobigbird-bert-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
length = 512

config=AutoConfig.from_pretrained(model_path)
config._name_or_path = 'kr.kim'
print(config)

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
    model = BBClassifier(model_path, config, dr_rate=0.5).to(device)
    earlystopping = EarlyStopping(path='./model/bb_' + str(fold) + 'fold_model.pt', patience=15, verbose=True, delta=0)

    kfold_train_dataloader, kfold_val_dataloader = kobigbird_make_fold(train_idx, val_idx, train_data, batch_size, tokenizer, length)

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
        train_loss, train_acc, scheduler = kobigbird_training(model, kfold_train_dataloader, optimizer, device, loss_fn, scheduler, fold, max_grad_norm, e)
        val_loss, val_acc = kobigbird_validate(model, kfold_val_dataloader, device, loss_fn, fold, e)

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