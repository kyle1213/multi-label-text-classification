import torch.utils.data
from training import kobigbird_Dataset, BBClassifier
from data import data_prep
from transformers import AutoTokenizer, AutoConfig
from tqdm import tqdm_notebook
from utils import *


device = connect_cuda()

model_path = "monologg/kobigbird-bert-base"
config = AutoConfig.from_pretrained(model_path)
config._name_or_path = 'kr.kim'

tokenizer = AutoTokenizer.from_pretrained(model_path)
length = 512
max_len = length
batch_size = 16


train_data, test_data = data_prep()
test_Dataset = kobigbird_Dataset(test_data, tokenizer, max_len)
test_dataloader = torch.utils.data.DataLoader(test_Dataset, batch_size=16, shuffle=False)
model = BBClassifier(dr_rate=0.5, config=config, model_path=model_path).to(device)

for fold in range(5):
    model.load_state_dict(torch.load('./model/bb_' + str(fold) + 'fold_model.pt'))
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

    print('micro: ', micro_calculate_f1_score(test_inference, targets))
    print('samples: ', samples_calculate_f1_score(test_inference, targets))
    print('macro: ', macro_calculate_f1_score(test_inference, targets))
    print('weighted: ', weighted_calculate_f1_score(test_inference, targets))
    print('acc: ', acc(test_inference, targets))
    print('one_acc: ', calculate_one_accuracy(test_inference, targets))
    print('zero_acc: ', calculate_zero_accuracy(test_inference, targets))
    print('50%-partial acc: ', partial_acc(test_inference, targets, 0.5))
    print('80%-partial acc: ', partial_acc(test_inference, targets, 0.8))
    print('90%-partial acc: ', partial_acc(test_inference, targets, 0.9))
    print('95%-partial acc: ', partial_acc(test_inference, targets, 0.95))
    print(print(f1_report(test_inference, targets)))

    for i in range(33):
        print(i+1, calculate_class_accuracy(test_inference, targets, i))