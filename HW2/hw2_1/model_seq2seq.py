import multiprocessing
import torch.optim as optim
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
import os
from scipy.special import expit
import random
import sys
import json
import re
from torch.utils.data import DataLoader, Dataset
import pickle
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
import model
data_path='MLDS_hw2_1_data/'
model_path='Model'
pickel_file='Model/picket_data.pickle'
class Dataprocessor(Dataset):
    def __init__(self, label_file, files_dir, dictonary, w2i):
        # def __init__(self):
        self.label_file = label_file
        self.files_dir = files_dir
        self.avi = filesreader(label_file)
        self.w2i = w2i
        self.dictonary = dictonary
        self.data_pair = helper1(files_dir, dictonary, w2i)
    def __len__(self):
        return len(self.data_pair)
    def __getitem__(self, idx):
        assert (idx < self.__len__())
        avi_file_name, sentence = self.data_pair[idx]
        data = torch.Tensor(self.avi[avi_file_name])
        data += torch.Tensor(data.size()).random_(0, 2000) / 10000
        return torch.Tensor(data), torch.Tensor(sentence)
class test_dataloader(Dataset):
    def __init__(self, test_data_path):
        self.avi = []
        files = os.listdir(test_data_path)
        for file in files:
            key = file.split('.npy')[0]
            value = np.load(os.path.join(test_data_path, file))
            self.avi.append([key, value])
    def __len__(self):
        return len(self.avi)
    def __getitem__(self, idx):
        return self.avi[idx]
def dictonaryFunc(word_min):
    with open(
            data_path+'training_label.json',
            'r') as f:
        file = json.load(f)
    wc = {}
    for d in file:
        for s in d['caption']:
            ws = re.sub('[.!,;?]]', ' ', s).split()
            for word in ws:
                word = word.replace('.', '') if '.' in word else word
                if word in wc:
                    wc[word] += 1
                else:
                    wc[word] = 1
    dict_1 = {}
    for word in wc:
        if wc[word] > word_min:
            dict_1[word] = wc[word]
    useful_tokens = [('<PAD>', 0), ('<SOS>', 1), ('<EOS>', 2), ('<UNK>', 3)]
    i2w = {i + len(useful_tokens): w for i, w in enumerate(dict_1)}
    w2i = {w: i + len(useful_tokens) for i, w in enumerate(dict_1)}
    for token, index in useful_tokens:
        i2w[index] = token
        w2i[token] = index
    return i2w, w2i, dict_1
def string_split(sentence, dictonary, w2i):  # sentenceSplit
    sentence = re.sub(r'[.!,;?]', ' ', sentence).split()
    for i in range(len(sentence)):
        if sentence[i] not in dictonary:
            sentence[i] = 3
        else:
            sentence[i] = w2i[sentence[i]]
    sentence.insert(0, 1)
    sentence.append(2)
    return sentence
def helper1(label_file, dictonary, w2i):
    label_json = label_file
    annotated_caption = []
    with open(label_json, 'r') as f:
        label = json.load(f)
    for d in label:
        for s in d['caption']:
            s = string_split(s, dictonary, w2i)
            annotated_caption.append((d['id'], s))
    return annotated_caption
def helper2(w2i, w):
    return w2i[w]

def helper3(i2w, i):
    return i2w[i]

def helper4(w2i, sentence):
    return [w2i[w] for w in sentence]

def helper5(i2w, index_seq):
    return [i2w[int(i)] for i in index_seq]
def filesreader(files_dir):
    avi_data = {}
    training_feats = files_dir
    files = os.listdir(training_feats)
    for file in files:
        value = np.load(os.path.join(training_feats, file))
        avi_data[file.split('.npy')[0]] = value
    return avi_data
def train(model, epoch, train_loader, loss_func):
    model.train()
    print(epoch)
    model = model
    parameters = model.parameters()
    optimizer = optim.Adam(parameters, lr=1e-4)
    #scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10)
    running_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        avi_feats, ground_truths, lengths = batch
        avi_feats, ground_truths = avi_feats, ground_truths
        avi_feats, ground_truths = Variable(avi_feats), Variable(ground_truths)

        optimizer.zero_grad()
        seq_logProb, seq_predictions = model(avi_feats, target_sentences=ground_truths, mode='train', tr_steps=epoch)

        ground_truths = ground_truths[:, 1:]
        loss = loss_cal(seq_logProb, ground_truths, lengths, loss_func)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 10 == 9:
            print(f"Batch: {batch_idx+1}/{len(train_loader)}, Loss: {running_loss/10:.3f}")
            running_loss = 0.0
        #scheduler.step(loss)
    loss = loss.item()
    print(f'Epoch:{epoch} & loss:{np.round(loss, 3)}')
def evaluate(test_loader, model):
    model.eval()
    for batch_idx, batch in enumerate(test_loader):
        val1, val2, lengths = batch
        val1, val2 = val1, val2
        val1, val2 = Variable(val1), Variable(val2)
        seq_logProb, seq_predictions = model(val1, mode='inference')
        val2 = val2[:, 1:]
        test_predictions = seq_predictions[:3]
        test_truth = val2[:3]
        break
def testfun(test_loader, model, i2w):
    model.eval()
    ss = []
    for batch_idx, batch in enumerate(test_loader):
        id, f1 = batch
        id, f1 = id, Variable(f1).float()
        seq_logProb, seq_predictions = model(f1, mode='inference')
        test_predictions = seq_predictions
        res = [[i2w[x.item()] if i2w[x.item()] != '<UNK>' else 'something' for x in s] for s in test_predictions]
        res = [' '.join(s).split('<EOS>')[0] for s in res]

        rr = zip(id, res)
        for r in rr:
            ss.append(r)
    return ss
def loss_cal(x, y, lengths, loss_fn):
    bs = len(x)
    p_cat = None
    g_cat = None
    flag = True
    for batch in range(bs):
        predict = x[batch]
        ground_truth = y[batch]
        seq_len = lengths[batch] - 1
        predict = predict[:seq_len]
        ground_truth = ground_truth[:seq_len]
        if flag:
            p_cat = predict
            g_cat = ground_truth
            flag = False
        else:
            p_cat = torch.cat((p_cat, predict), dim=0)
            g_cat = torch.cat((g_cat, ground_truth), dim=0)
    loss = loss_fn(p_cat, g_cat)
    avg_loss = loss / bs
    return loss
def minibatch(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    avi_data, captions = zip(*data)
    avi_data = torch.stack(avi_data, 0)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return avi_data, targets, lengths
def main():
    label_file = data_path+'training_data/feat'
    files_dir = data_path+'training_label.json'
    i2w,w2i,dictonary = dictonaryFunc(4)
    train_dataset = Dataprocessor(label_file, files_dir,dictonary, w2i)
    train_dataloader = DataLoader(dataset = train_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=minibatch)
    
    label_file = data_path+'testing_data/feat'
    files_dir = data_path+'testing_label.json'
    test_dataset = Dataprocessor(label_file,files_dir,dictonary, w2i)
    test_dataloader = DataLoader(dataset = test_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=minibatch)
   
    epochs_n = 20
    ModelSaveLoc = (model_path)
    with open(pickel_file, 'wb') as f:
         pickle.dump(i2w, f)

    
    x = len(i2w)+4
    if not os.path.exists(ModelSaveLoc):
        os.mkdir(ModelSaveLoc)
    loss_fn = nn.CrossEntropyLoss()
    encode =model.EncoderNet()
    decode = model.DecoderNet(512, x, x, 1024, 0.3)
    modeltrain = model.ModelMain(encoder = encode,decoder = decode) 
    

    start = time.time()
    for epoch in range(epochs_n):
        train(modeltrain,epoch+1, train_loader=train_dataloader, loss_func=loss_fn)
        evaluate(test_dataloader, modeltrain)

    end = time.time()
    torch.save(modeltrain, "{}/{}.h5".format(ModelSaveLoc, 'model'))
    print("Training finished {}  elapsed time: {: .3f} seconds. \n".format('test', end-start))

if __name__=="__main__":
    if sys.platform.startswith('win'):
        multiprocessing.freeze_support()
    main()
