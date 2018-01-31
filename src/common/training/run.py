import json

import torch
import torch.nn.functional as F
from sklearn.utils import shuffle

from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from common.training.batcher import Batcher, prepare, prepare_with_labels
from common.util.random import SimpleRandom


def evaluate(model,data,labels,batch_size):
    predicted = predict(model,data,batch_size)
    return accuracy_score(labels,predicted.data.numpy().reshape(-1))

def predict(model, data, batch_size):
    batcher = Batcher(data, batch_size)

    predicted = []
    for batch, size, start, end in batcher:
        d = prepare(batch)
        model.eval()
        logits = model(d).cpu()

        predicted.extend(torch.max(logits, 1)[1])
    return torch.stack(predicted)

def train(model, fs, batch_size, lr, epochs,dev=None, clip=None, early_stopping=None,name=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    data, labels = fs
    if dev is not None:
        dev_data,dev_labels = dev

    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        epoch_data = 0

        shuffle(data,labels)

        batcher = Batcher(data, batch_size)

        for batch, size, start, end in batcher:
            d,gold = prepare_with_labels(batch,labels[start:end])

            model.train()
            optimizer.zero_grad()
            logits = model(d)

            loss = F.cross_entropy(logits, gold)
            loss.backward()

            epoch_loss += loss.cpu()
            epoch_data += size

            if clip is not None:
                torch.nn.utils.clip_grad_norm(model.parameters(), clip)
            optimizer.step()

        print("Average epoch loss: {0}".format((epoch_loss/epoch_data).data.numpy()))

        #print("Epoch Train Accuracy {0}".format(evaluate(model, data, labels, batch_size)))
        if dev is not None:
            acc = evaluate(model,dev_data,dev_labels,batch_size)
            print("Epoch Dev Accuracy {0}".format(acc))

            if early_stopping is not None and early_stopping(model,acc):
                break

    if dev is not None and early_stopping is not None:
        early_stopping.set_best_state(model)



def print_evaluation(model,data,ls,log=None):
    features,actual = data
    predictions = predict(model, features, 500).data.numpy().reshape(-1).tolist()

    labels = [ls.idx[i] for i, _ in enumerate(ls.idx)]

    actual = [labels[i] for i in actual]
    predictions = [labels[i] for i in predictions]

    print(accuracy_score(actual, predictions))
    print(classification_report(actual, predictions))
    print(confusion_matrix(actual, predictions))

    data = zip(actual,predictions)
    if log is not None:
        f = open(log, "w+")
        for a,p in data:
            f.write(json.dumps({"actual": a, "predicted": p}) + "\n")
        f.close()
