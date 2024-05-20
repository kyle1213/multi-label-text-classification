from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from sklearn.model_selection import KFold
from data import *
from utils import *
from train_module import AsymmetricLoss, EarlyStopping, BBClassifier, train_model, validate_model, make_fold


device = connect_cuda()

config = {
    'data_prep': 'shoes',
    'pretrained_model_path': "monologg/kobigbird-bert-base",
    'max_length_token': 512,
    'batch_size': 16,
    'warmup_ratio': 0.1,
    'num_epochs': 100,
    'max_grad_norm': 1,
    'log_interval': 200,
    'learning_rate': 5e-5,
    'k_fold_N': 5,
    'k_fold_shuffle': True,
    'num_classes': 33,
    'dr_rate': 0.1,
    'model_save_path': './model/',
    'model_save_name': 'bb_',
    'early_stop_patience': 3,
    'early_stop_delta': 0,
    'weight_decay': 0.01,
    'no_weight_decay': 0.00,
    'loss_fn': AsymmetricLoss(),
    'optimizer': AdamW,
}

train_data, test_data = shoes_data_prep() if config['data_prep'] == 'shoes' else None
print(len(train_data), len(test_data))

tokenizer = AutoTokenizer.from_pretrained(config['pretrained_model_path'])
model_config = AutoConfig.from_pretrained(config['pretrained_model_path'])
model_config._name_or_path = 'kr.kim'
print(model_config)

kfold = KFold(n_splits=config['k_fold_N'],
              shuffle=config['k_fold_shuffle'],
              random_state=0)
kfolds = kfold.split(train_data)

train_losses_list_per_fold = []
train_accs_per_fold = []
val_losses_list_per_fold = []
val_accs_per_fold = []

for fold, (train_idx, val_idx) in enumerate(kfolds):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    if config['pretrained_model_path'] == "monologg/kobigbird-bert-base":
        model = BBClassifier(model_path=config['pretrained_model_path'],
                             config=model_config,
                             num_classes=config['num_classes'],
                             dr_rate=config['dr_rate']).to(device)
    else:  # elif config['pretrained_model_path'] == "beomi/KcELECTRA-base-v2022:
        model = AutoModelForSequenceClassification.from_pretrained('beomi/KcELECTRA-base-v2022', num_labels=33, problem_type='multi_label_classification').to(device)
    earlystopping = EarlyStopping(path=str(config['model_save_path']) + str(config['model_save_name']) + str(fold) + 'fold_model.pt',
                                  patience=config['early_stop_patience'],
                                  verbose=True,
                                  delta=config['early_stop_delta'])

    kfold_train_dataloader, kfold_val_dataloader = make_fold(model_path=config['pretrained_model_path'],
                                                             train_idx=train_idx,
                                                             val_idx=val_idx,
                                                             train_data=train_data,
                                                             batch_size=config['batch_size'],
                                                             tokenizer=tokenizer,
                                                             length=config['max_length_token'])

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': config['weight_decay']},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': config['no_weight_decay']}
    ]
    optimizer = config['optimizer'](optimizer_grouped_parameters,
                                    lr=config['learning_rate'])
    loss_fn = config['loss_fn']

    t_total = len(kfold_train_dataloader) * config['num_epochs']
    warmup_step = int(t_total * config['warmup_ratio'])
    scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=warmup_step,
                                                num_training_steps=t_total)

    for e in range(config['num_epochs']):
        train_loss, train_acc, scheduler = train_model(model_path=config['pretrained_model_path'],
                                                       model=model,
                                                       kfold_train_dataloader=kfold_train_dataloader,
                                                       optimizer=optimizer,
                                                       device=device,
                                                       loss_fn=loss_fn,
                                                       scheduler=scheduler,
                                                       fold=fold,
                                                       max_grad_norm=config['max_grad_norm'],
                                                       e=e)
        val_loss, val_acc = validate_model(model_path=config['pretrained_model_path'],
                                           model=model,
                                           kfold_val_dataloader=kfold_val_dataloader,
                                           device=device,
                                           loss_fn=loss_fn,
                                           fold=fold,
                                           e=e)

        earlystopping(val_loss=val_loss,
                      model=model)
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

for i in range(5):
    plt.plot(train_accs_per_fold[i])
    plt.plot(val_accs_per_fold[i])
    plt.show()
    plt.clf()

for i in range(5):
    plt.plot(train_losses_list_per_fold[i])
    plt.plot(val_losses_list_per_fold[i])
    plt.show()
    plt.clf()