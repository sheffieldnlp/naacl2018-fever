import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from common.training.batcher import Batcher, prepare, prepare_with_labels

import torch.nn.functional as F

def evaluate(model,data,labels,batch_size):
    predicted = predict(model,data,batch_size)
    return accuracy_score(labels,predicted.data.numpy().reshape(-1))

def predict(model, data, batch_size):
    batcher = Batcher(data, batch_size)

    predicted = []
    for batch, size, start, end in batcher:
        d = prepare(batch)

        logits = model(d).cpu()

        predicted.extend(torch.max(logits, 1)[1])
    return torch.stack(predicted)

def train(model, fs, batch_size, lr, epochs,dev=None):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    steps = 0

    data, labels = fs

    data = data
    labels = labels
    if dev is not None:
        dev_data,dev_labels = dev


    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        epoch_data = 0

        batcher = Batcher(data, batch_size)

        for batch, size, start, end in batcher:
            d,gold = prepare_with_labels(batch,labels[start:end])

            optimizer.zero_grad()
            logits = model(d)

            loss = F.cross_entropy(logits, gold)
            loss.backward()

            epoch_loss += loss.cpu()
            epoch_data += size
            optimizer.step()

        print("Average epoch loss: {0}".format((epoch_loss/epoch_data).data.numpy()))

        print("Epoch Train Accuracy {0}".format(evaluate(model, data, labels, batch_size)))
        if dev is not None:
            print("Epoch Dev Accuracy {0}".format(evaluate(model,dev_data,dev_labels,batch_size)))