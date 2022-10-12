'''
Tips:
========================================================================================
* The main code for BRNet training, validating and testing
* It is such a great pity that the dataset of palm-leaf manuscripts would not be available
  because we have signed a separate confidentiality agreement.
* Parameters in the following code should be adjusted to adapt to in your own case.
* It is worth mentioning that the value of batch_size depends on memeory size, if you train 
  the network on CPU (Actually we don't suggest you to train in this way due to a significant
  improvement brought by training on GPU), it may cost plenty of time.
========================================================================================
'''

import torch
import torch.nn as nn
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from Architecture.BRNet_Architecture import UNet, UNet1, UNet_4l
from Dataset.Dataset import Mydataset
import matplotlib.pyplot as plt

from Main.Metrics import SegmentationMetric


class BRNet(nn.Module):
    def __init__(self, model, device):
        super().__init__()
        self.device = device
        self.model = model
        self.metric = SegmentationMetric(2)

    def train_network(self, learning_rate, weight_decay, train_val_ratio, epochs, batch_size):
        '''
        this function is used for BRNet training.
        the training set is used to train the network.
        the validation set is used to choose the hyperparameters.
        
        Parameters
        ----------
        learning_rate : float
            e.g. 0.001 or 1e-3
            the learning rate for BRNet training.
        weight_decay : float
            e.g. 0.001 or 1e-3
            weight decay applied to BRNet training to alleviate over-fitting problem.
            weight decay is actually an optional operation, if not necessary, the value
            of weight decay can be set as 0.
        train_val_ratio : float
            e.g. 0.8
            the ratio of the training set and the validation set.
            the value is suggested to be set as 0.7 or 0.8.
        epochs : int
            e.g. 300
            the times of BRNet training.
        batch_size : int
            e.g. 4
            the size of each batch fed into the network.
            set a suitable batch size to make full use of memory

        Returns
        -------
        epoch_train_loss_list : list
            a list of loss of training set for each epoch of training.
        epoch_train_miou_list : list
            a list of miou of training set for each epoch of training.
        epoch_val_loss_list : list
            a list of loss of validation set for each epoch of training.
        epoch_val_miou_list : list
            a list of miou of validation set for each epoch of training.
        '''
        epoch_train_loss_list = []
        epoch_train_miou_list = []
        epoch_val_loss_list = []
        epoch_val_miou_list = []
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=101, gamma=0.1)
        criterion = torch.nn.CrossEntropyLoss()
        
        dataset = Mydataset("patches_path_61")
        train_size = int(len(dataset) * train_val_ratio)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_dataload = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        validation_dataload = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            # initialize all parameters
            epoch_train_loss = 0
            epoch_train_miou = 0
            epoch_val_loss = 0
            epoch_val_miou = 0
            step = 0
            self.model.train()
            for x, y in train_dataload:
                self.model.train()
                inputs = x.to(self.device)
                outputs = self.model(inputs)
                labels = y.to(self.device)
                labels = labels.long().squeeze(1)
                loss = criterion(outputs, labels)
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # exp_lr_scheduler.step()
        
                epoch_train_loss += loss.item()
                step += 1
                if step % 70 == 0:
                    print('epoch %d step %d loss:%0.6f' % (epoch, step, loss.item()))
                
                py = outputs.detach().cpu().numpy()
                a = np.argmax(py, axis=1)
                self.metric.addBatch(y.squeeze(1).data.numpy().astype(np.int64), a)
                train_m_miou = self.metric.meanIntersectionOverUnion()[0]
                epoch_train_miou += train_m_miou
            current_train_miou = epoch_train_miou / len(train_dataload)
            epoch_train_miou_list.append(current_train_miou)
            print('epoch %d training miou:%0.4f' % (epoch, current_train_miou))

            current_train_loss = epoch_train_loss / len(train_dataload)
            epoch_train_loss_list.append(current_train_loss)
            print('epoch %d training loss:%0.5f' % (epoch, current_train_loss))

            self.model.eval()
            with torch.no_grad():
                for x1, y1 in validation_dataload:
                    inputs = x1.to(self.device)
                    outputs = self.model(inputs)
                    labels = y1.to(self.device)
                    labels = labels.long().squeeze(1)
                    loss = criterion(outputs, labels)

                    epoch_val_loss += loss.item()

                    py = outputs.cpu().numpy()
                    a = np.argmax(py, axis=1)
                    self.metric.addBatch(y1.squeeze(1).data.numpy().astype(np.int64), a)
                    val_m_miou = self.metric.meanIntersectionOverUnion()[0]
                    epoch_val_miou += val_m_miou
                current_val_miou = epoch_val_miou / len(validation_dataload)
                epoch_val_miou_list.append(current_val_miou)
                print('epoch %d validation miou:%0.4f' % (epoch, current_val_miou))

                current_val_loss = epoch_val_loss / len(validation_dataload)
                epoch_val_loss_list.append(current_val_loss)
                print('epoch %d validation loss:%0.5f' % (epoch, current_val_loss))
        return epoch_train_loss_list, epoch_train_miou_list, epoch_val_loss_list, epoch_val_miou_list
    
    def test_network(self):
        '''
        this function is used for BRNet testing.
        
        Returns
        -------
        m_precision : float
            the mean pixel accuracy of BRNet performing on test set.
        miou : float
            the mean intersection over union of BRNet performing on test set.
        m_recall : float
            the mean recall of BRNet performing on test set.
        Micro_F1_score : float
            the Micro f1-score of BRNet performing on test set.

        '''

        test_dataset = Mydataset("patches_path_9")
        test_size = len(test_dataset)
        test_dataload = DataLoader(test_dataset, batch_size=1, shuffle=True)
        with torch.no_grad():
            for x_, y_ in test_dataload:
                self.model.eval()
                inputs = x_.to(self.device)
                outputs = self.model(inputs)  # 前向传播
                py = outputs.cpu().numpy()
                py = py.reshape(2, 512, 400)
                a = np.argmax(py, axis=0)
        
                self.metric.addBatch(y_.data.numpy().reshape(512, 400).astype(np.int64), a)
            m_precision = self.metric.meanPixelAccuracy()
            m_recall = self.metric.meanRecall()
            Micro_F1_score = self.metric.F1_score()
            miou = self.metric.meanIntersectionOverUnion()[0]
        return m_precision, miou, m_recall, Micro_F1_score
    
    def save_network(self, model_name):
        torch.save(self.model.state_dict(), '{}.pkl'.format(model_name))


if __name__ == '__main__':
    # instantiation of the class BRNet, the allocated model is typical BRNet with 10 layers
    print("Preparation for BRNet training...")
    print("Model construction and BRNet trainer loading...")
    device = torch.device("cuda:0")
    model = UNet(3, 2).to(device)
    BRNet_Trainer = BRNet(model, device)
    
    # initialization of parameters for BRNet training
    print("Parameters for BRNet training is setting...")
    l_r = 1e-6
    weight_decay = 1e-6
    train_val_ratio = 0.8
    epochs = 250
    batch_size = 4
    
    # BRNet training
    print("BRNet training begin...")
    losses_mious = BRNet_Trainer.train_network(l_r,
                                               weight_decay,
                                               train_val_ratio,
                                               epochs,
                                               batch_size)
    
    epoch_train_loss_list = np.array(losses_mious[0])
    epoch_train_miou_list = np.array(losses_mious[1])
    epoch_val_loss_list = np.array(losses_mious[2])
    epoch_val_miou_list = np.array(losses_mious[3])
    
    import os
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_ylim(0, epoch_train_loss_list.max() * 1.1)
    
    lns1 = ax.plot(epoch_train_loss_list, label="train Loss")
    lns2 = ax.plot(epoch_val_loss_list, label="validation Loss")
    ax2 = ax.twinx()
    lns3 = ax2.plot(epoch_train_miou_list, c='green', label="train MIoU")
    lns4 = ax2.plot(epoch_val_miou_list, c='r', label="validation MIoU")
    
    lns = lns1+lns2+lns3+lns4
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='center right')
    
    ax.set_xlabel("Epoch (Times)")
    ax.set_ylabel("Loss")
    ax2.set_ylabel("MIoU (%)")
    ax2.set_ylim(epoch_train_miou_list.min() - 0.01, 1.00)
    
    plt.savefig("curves_{}.png".format("model"))
    plt.close()
    
    print("BRNet training finish. Save the lists of loss and miou of training set and validation set...")
    np.save("train_loss_{}.npy".format("model"), epoch_train_loss_list)
    np.save("val_loss_{}.npy".format("model"), epoch_val_loss_list)
    np.save("train_miou_{}.npy".format("model"), epoch_train_miou_list)
    np.save("val_miou_{}.npy".format("model"), epoch_val_miou_list)
    
    # BRNet testing
    print("BRNet testing begin...")
    test_mpa, test_miou, test_mrecall, test_f1score = BRNet_Trainer.test_network()
    print('validation mpa:%0.4f, miou:%0.4f, mrecall:%0.4f, Micro f1-score:%0.4f' % (test_mpa, test_miou, test_mrecall, test_f1score))
    
    # Save model
    print("Saving BRNet model...")
    BRNet_Trainer.save_network("model")