import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm_notebook
from sklearn.metrics import f1_score

from data import data_prep
from utils import connect_cuda, acc, calculate_one_accuracy, calculate_zero_accuracy, calculate_class_accuracy


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


device = connect_cuda()

train_data, test_data = data_prep()

tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base-v2022")
length = 512

test_Dataset = CustomDataset(test_data, tokenizer)
test_dataloader = torch.utils.data.DataLoader(test_Dataset, batch_size=1, shuffle=False)
model = AutoModelForSequenceClassification.from_pretrained('beomi/KcELECTRA-base-v2022', num_labels=33, problem_type='multi_label_classification').to(device)

targets_lists = []
test_inference_lists = []
for fold in range(5):
    model.load_state_dict(torch.load('./model/kcelectra_' + str(fold) + 'fold_model.pt'))
    test_inference = []
    targets = []
    model.eval()
    with torch.no_grad():
        for batch_id, (input_ids_batch, attention_masks_batch, token_type_ids_batch, y_batch) in enumerate(tqdm_notebook(test_dataloader)):
            input_ids_batch = input_ids_batch.long().to(device)
            attention_masks_batch = attention_masks_batch.long().to(device)
            token_type_ids_batch = token_type_ids_batch.long().to(device)
            y_batch = y_batch.long().to(device)
            out = model(input_ids_batch, attention_masks_batch, token_type_ids_batch)[0]
            out_onehot = torch.where(out < 0, torch.tensor(0), torch.tensor(1))
            test_inference.append(out_onehot.cpu()[0])
            targets.append(y_batch.cpu()[0])

        test_inference_lists.append(test_inference)
        targets_lists.append(targets)

    print('micro: ', f1_score(test_inference, targets, average='micro'))
    print('samples: ', f1_score(test_inference, targets, average='samples'))
    print('macro: ', f1_score(test_inference, targets, average='macro'))
    print('weighted: ', f1_score(test_inference, targets, average='weighted'))
    print('acc: ', acc(test_inference, targets))
    print('one_acc: ', calculate_one_accuracy(test_inference, targets))
    print('zero_acc: ', calculate_zero_accuracy(test_inference, targets))

    for i in range(33):
        print(i+1, calculate_class_accuracy(test_inference, targets, i))

ensemble = []
for a, b, c, d, e in zip(test_inference_lists[0], test_inference_lists[1], test_inference_lists[2], test_inference_lists[3], test_inference_lists[4]):
    x = a+b+c+d+e
    ensemble.append(x)

print(len(ensemble))

for i, t in enumerate(ensemble):
    ensemble[i] = torch.where(t > 2.5, torch.tensor(1), torch.tensor(0))

y_labels = []
for i in range(len(ensemble)):
    y_labels.append(targets_lists[0][i])

print('micro: ', f1_score(ensemble, y_labels, average='micro'))
print('samples: ', f1_score(ensemble, y_labels, average='samples'))
print('macro: ', f1_score(ensemble, y_labels, average='macro'))
print('weighted: ', f1_score(ensemble, y_labels, average='weighted'))
print('acc: ', acc(ensemble, y_labels))
print('one_acc: ', calculate_one_accuracy(ensemble, y_labels))
print('zero_acc: ', calculate_zero_accuracy(ensemble, y_labels))

for i in range(33):
    print(i + 1, calculate_class_accuracy(ensemble, y_labels, i))