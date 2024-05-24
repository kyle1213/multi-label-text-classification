import torch.utils.data
from train_module import Dataset, BBClassifier
from data import shoes_data_prep, bhc_data_prep
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


train_data, test_data = bhc_data_prep()
test_Dataset = Dataset(model_path, test_data, tokenizer, max_len)
test_dataloader = torch.utils.data.DataLoader(test_Dataset, batch_size=16, shuffle=False)
model = BBClassifier(num_classes=31, config=config, model_path=model_path).to(device)

test_inferences = []
for fold in range(5):
    model.load_state_dict(torch.load('./model/bhc_bb_' + str(fold) + 'fold_model.pt'))
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

    for i in range(31):
        print(i+1, calculate_class_accuracy(test_inference, targets, i))

    test_inferences.append(test_inference)


def ensemble_inference(t_):
    ensemble = []
    for a, b, c, d, e in zip(test_inferences[0], test_inferences[1], test_inferences[2], test_inferences[3], test_inferences[4]):
        x = a+b+c+d+e
        ensemble.append(x)

    for i, t in enumerate(ensemble):
        ensemble[i] = torch.where(t > float(t_), torch.tensor(1), torch.tensor(0))

    y_labels = targets

    print(ensemble[0])
    print(y_labels[0])

    print('micro: ', micro_calculate_f1_score(ensemble, y_labels))
    print('samples: ', samples_calculate_f1_score(ensemble, y_labels))
    print('macro: ', macro_calculate_f1_score(ensemble, y_labels))
    print('weighted: ', weighted_calculate_f1_score(ensemble, y_labels))
    print('acc: ', acc(ensemble, y_labels))
    print('one_acc: ', calculate_one_accuracy(ensemble, y_labels))
    print('zero_acc: ', calculate_zero_accuracy(ensemble, y_labels))
    print('50%-partial acc: ', partial_acc(ensemble, y_labels, 0.5))
    print('80%-partial acc: ', partial_acc(ensemble, y_labels, 0.8))
    print('90%-partial acc: ', partial_acc(ensemble, y_labels, 0.9))
    print('95%-partial acc: ', partial_acc(ensemble, y_labels, 0.95))
    print(print(f1_report(ensemble, y_labels)))

    for i in range(31):
        print(i+1, calculate_class_accuracy(ensemble, y_labels, i))


t_list = [0.5, 1.5, 2.5, 3.5, 4.5]

for t in t_list:
    ensemble_inference(t)
    print("-------------------------------------------------------------------")