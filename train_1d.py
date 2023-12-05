import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from models.sequence.model_1d import TemporalSequenceModel
from sklearn.model_selection import train_test_split
from models.sequence.sequence_loader import Loader
import sys

sys.path.append("../../")
import numpy as np
import os

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

torch.manual_seed(1111)



temporal_structure = {1: [64, 2, False],
                    2: [64, 2, True],
                    }
train_val_ratio = 0.3


class SequenceTemporalTrainer:
    def __init__(self, data_path, epoch, dropout, lr, batch_size, temporal_module, exp_folder, n_classes, struct_num,
                 load_model="", evaluate=False):
        self.epoch = epoch
        self.batch_size = batch_size
        [hidden_dims, num_rnn_layers, attention] = temporal_structure[struct_num]
        self.model = TemporalSequenceModel(num_classes=n_classes, input_dim=input_channels, hidden_dims=hidden_dims,
                                          num_rnn_layers=num_rnn_layers, attention=attention, temporal_module=temporal_module,
                                           dropout=dropout, struct_num=struct_num)
        if device != 'cpu':
            self.model.cuda()

        os.makedirs(exp_folder, exist_ok=True)
        self.name = os.path.join(exp_folder, "model.pth")
        self.log = open(os.path.join(exp_folder, "result.txt"), 'w')
        self.criterion = torch.nn.CrossEntropyLoss().to(device=device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
        self.data, self.label = self.__load_data(data_path)
        sample = [(d, l) for d, l in zip(self.data, self.label)]
        train_sample, test_sample = train_test_split(sample, test_size=train_val_ratio, random_state=1,
                                                     shuffle=True)
        train_data, train_labels = self.__separate_sample(train_sample)
        test_data, test_labels = self.__separate_sample(test_sample)
        train_set = Loader(train_data, train_labels, n_classes)
        valid_set = Loader(test_data, test_labels, n_classes)
        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        self.valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, drop_last=True)
        self.evaluate = evaluate
        assert not (self.evaluate and load_model == ""), "You must load model when you want to evaluate"

    def __load_data(self, root):
        with open(os.path.join(root, "data.txt"), "r") as data_f:
            data = []
            for line in data_f.readlines():
                origin_ls = self.__ls_preprocess(line.split("\t"))
                ls = [float(item) for item in origin_ls]
                data.append(np.array(ls).reshape((seq_length, input_channels)))
            data_f.close()

        with open(os.path.join(root, "label.txt"), "r") as label_f:
            label = [int(line[:-1]) for line in label_f.readlines()]
            label_f.close()
        return data, label

    @staticmethod
    def __separate_sample(sample):
        data, label = [], []
        for item in sample:
            data.append(item[0])
            label.append(item[1])
        return data, label

    @staticmethod
    def __ls_preprocess(ls):
        try:
            ls.remove("\n")
        except:
            pass
        while True:
            try:
                ls.remove("")
            except ValueError:
                break
        return ls

    def __train(self, ep):
        global steps
        train_loss = 0
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(device=device)
            target = target.to(device=device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss
            steps += seq_length
            if batch_idx > 0 and batch_idx % log_interval == 0:
                out_log = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                    ep, batch_idx * self.batch_size, len(self.train_loader.dataset),
                        100. * batch_idx / len(self.train_loader), train_loss.item() / log_interval, steps)
                print(out_log)
        return train_loss

    def __test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.valid_loader:
                data, target = data.to(device=device), target.to(device=device)
                total += target.size(0)
                output = self.model(data)
                pred = torch.max(output, 1)[1]
                test_loss += self.criterion(output, target)
                correct += pred.eq(target).sum().item()
            test_loss /= len(self.valid_loader.dataset)
            out_log = 'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, total,
                100. * correct / total)
            print(out_log)
            self.log.write(out_log + '\n')
        return test_loss, 100. * correct / total

    def run(self):
        if self.evaluate:
            self.model.load_state_dict(torch.load(self.name))
            self.model.eval()
            test_loss, test_acc = self.__test()
            return test_loss, test_acc

        min_val_loss, min_train_loss, max_val_acc = float("inf"), float("inf"), 0
        for epoch in range(1, self.epoch + 1):
            train_loss = self.__train(epoch)
            min_train_loss = train_loss if train_loss < min_train_loss else min_train_loss
            val_loss, val_acc = self.__test()
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                torch.save(self.model.state_dict(), self.name)
                self.log.write("Model from {} epoch is saved\n".format(epoch))
            max_val_acc = max(max_val_acc, val_acc)
        return min_train_loss, min_val_loss, max_val_acc


if __name__ == '__main__':
    input_channels = 17 * 2
    log_interval = 5
    seq_length = 20
    device = "cpu"
    temporal_module = "TCN" #[BiLSTM, LSTM, BiGRU, GRU, TCN]
    exp_path = "exp/TCN"
    steps = 0

    SequenceTemporalTrainer("data/kps_data/input2/equal", 30, 0.05, 1e-4, 8,
                            temporal_module, exp_path, 2, 1).run()
    evaluate = True
    model_path = os.path.join(exp_path, "model.pth")
    SequenceTemporalTrainer("data/kps_data/input2/equal", 30, 0.05, 1e-4, 8,
                  temporal_module, exp_path, 2, 1, model_path, evaluate).run()