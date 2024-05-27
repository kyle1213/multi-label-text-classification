import torch.utils.data
from train_module import Dataset, BBClassifier
from data import data_prep
from transformers import AutoTokenizer, AutoConfig
from tqdm import tqdm_notebook
from utils import *


device = connect_cuda()

bhc_config = {
    'data_prep': 'bhc',
    'pretrained_model_path': "monologg/kobigbird-bert-base",
    'max_length_token': 512,
    'batch_size': 16,
    'num_classes': 31,
    'model_save_path': './model/',
    'model_save_name': 'bhc_bb_',
    'k_fold_N': 5
}

config = bhc_config

train_data, test_data = data_prep(data_name=config['data_prep'])

tokenizer = AutoTokenizer.from_pretrained(config['pretrained_model_path'])
model_config = AutoConfig.from_pretrained(config['pretrained_model_path'])
model_config._name_or_path = 'kr.kim'
print(model_config)

test_Dataset = Dataset(model_path=config['pretrained_model_path'], data=test_data, tokenizer=tokenizer, length=config['max_length_token'])
test_dataloader = torch.utils.data.DataLoader(test_Dataset, batch_size=config['batch_size'], shuffle=False)
model = BBClassifier(num_classes=config['num_classes'], config=model_config, model_path=config['pretrained_model_path']).to(device)

test_inferences = []
targets = []
for fold in range(config['k_fold_N']):
    model.load_state_dict(torch.load(str(config['model_save_path']) + str(config['model_save_name']) + str(fold) + 'fold_model.pt'))
    test_inference = []
    targets = []
    model.eval()
    with torch.no_grad():
        for batch_id, data in enumerate(tqdm_notebook(test_dataloader)):
            input_ids = data['input_ids'].long().to(device)
            attention_mask = data['attention_mask'].long().to(device)
            label = data['labels'].long().to(device)
            out = model(input_ids, attention_mask)
            out_onehot = torch.where(out < 0, torch.tensor(0), torch.tensor(1))
            for o in out_onehot:
                test_inference.append(o.cpu())
            for l in label:
                targets.append(l.cpu())

    metrics(targets, test_inference)

    for i in range(config['num_classes']):
        print(i+1, calculate_class_accuracy(test_inference, targets, i))

    test_inferences.append(test_inference)

t_list = [i + 0.5 for i in range(config['k_fold_N'])]
for t in t_list:
    ensemble = []
    tmp_list = [test_inferences[i] for i in range(config['k_fold_N'])]
    for x in zip(*tmp_list):
        x = sum(x)
        ensemble.append(x)

    for i, t_ in enumerate(ensemble):
        ensemble[i] = torch.where(t_ > float(t), torch.tensor(1), torch.tensor(0))

    y_labels = targets

    metrics(targets, ensemble)

    for i in range(config['num_classes']):
        print(i+1, calculate_class_accuracy(y_labels, ensemble, i))