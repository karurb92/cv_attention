#To be refactored
#Potential problem with forward pass on validation data

import torch
import numpy as np

class Solver(object):

    def __init__(self, model, train_dataloader, val_dataloader, device, patience):

        self.device = device
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.patience = patience
        
        self.batch_size = model.hparams['batch_size']
        self.learning_rate = model.hparams['learning_rate']
        self.epochs = model.hparams['epochs']
        self.loss_func = model.hparams['loss_func']
        self.optimizer = model.hparams['optimizer'](list(self.model.parameters()), lr=self.learning_rate)

        self.best_model_stats = None
        self.best_params = None
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.train_batch_loss = []
        self.val_batch_loss = []
        self.current_patience = 0

        self.train_TP_history = []
        self.train_FP_history = []
        self.train_TN_history = []
        self.train_FN_history = []
        self.val_TP_history = []
        self.val_FP_history = []
        self.val_TN_history = []
        self.val_FN_history = []


    def _step(self, images, labels, validation):
        self.optimizer.zero_grad()
        if validation:
            with torch.no_grad():
                predictions = self.model.forward(images)
            #loss = self.loss_func(predictions[:, 1].squeeze(), labels.float())    
            loss = self.loss_func(predictions.squeeze(), labels)
        else:
            predictions = self.model.forward(images)
            #loss = self.loss_func(predictions[:, 1].squeeze(), labels.float())
            loss = self.loss_func(predictions.squeeze(), labels)
            loss.backward()
            self.optimizer.step()
        return loss.item(), predictions


    def train(self):

        for epoch in range(self.epochs):
            
            ######################### Iterate over all training samples #########################
            train_epoch_loss = 0.0
            running_loss = 0.0
            train_TP = 0
            train_FP = 0
            train_TN = 0
            train_FN = 0

            for i, batch in enumerate(self.train_dataloader):

                if not batch: break
                
                images, labels = batch['image'], batch['label']
                images = images.to(self.device)
                labels = labels.to(self.device)
        
                train_loss, predictions = self._step(images, labels, validation=False)

                label_pred = torch.argmax(predictions, dim=1)
                for p, l in zip(label_pred, labels):
                    if p==1 and l==1: train_TP +=1
                    if p==1 and l==0: train_FP +=1
                    if p==0 and l==0: train_TN +=1
                    if p==0 and l==1: train_FN +=1

                train_epoch_loss += train_loss
                running_loss += train_loss
                self.train_batch_loss.append(train_loss)

                # Print statistics to console
                if i % 5 == 4: # print every 5 mini-batches
                    running_loss /= 5
                    print("[Epoch %d, Iteration %5d] loss: %.5f" % (epoch+1, i+1, running_loss))
                    running_loss = 0.0

            train_epoch_loss /= len(self.train_dataloader)
            self.train_loss_history.append(train_epoch_loss)
            print(f'Train loss after epoch {epoch+1}: {train_epoch_loss}')

            ######################### Iterate over all validation samples #########################
            val_epoch_loss = 0.0
            val_TP = 0
            val_FP = 0
            val_TN = 0
            val_FN = 0

            for batch in self.val_dataloader:

                if not batch: break

                images, labels = batch['image'], batch['label']
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Compute Loss - no param update at validation time!
                val_loss, predictions = self._step(images, labels, validation=True)

                label_pred = torch.argmax(predictions, dim=1)
                for p, l in zip(label_pred, labels):
                    if p==1 and l==1: val_TP +=1
                    if p==1 and l==0: val_FP +=1
                    if p==0 and l==0: val_TN +=1
                    if p==0 and l==1: val_FN +=1

                val_epoch_loss += val_loss
                self.val_batch_loss.append(val_loss)
                
            val_epoch_loss /= len(self.val_dataloader)
            self.val_loss_history.append(val_epoch_loss)
            print(f'Val loss after epoch {epoch+1}: {val_epoch_loss}')


            ######################### Report measures #########################
            train_epoch_acc = (train_TP+train_TN) / (train_TP+train_TN+train_FN+train_FP)
            val_epoch_acc = (val_TP+val_TN) / (val_TP+val_TN+val_FN+val_FP)
            self.train_acc_history.append(train_epoch_acc)
            self.val_acc_history.append(val_epoch_acc)

            self.train_TP_history.append(train_TP)
            self.train_FP_history.append(train_FP)
            self.train_TN_history.append(train_TN)
            self.train_FN_history.append(train_FN)
            self.val_TP_history.append(val_TP)
            self.val_FP_history.append(val_FP)
            self.val_TN_history.append(val_TN)
            self.val_FN_history.append(val_FN)
            #print(f'Training accuracy after epoch {epoch+1}: {train_epoch_acc}')
            #print(f'Validation accuracy after epoch {epoch+1}: {val_epoch_acc}')


            ######################### Keep track of the best model #########################
            self.update_best_loss(val_epoch_loss, train_epoch_loss)
            if self.patience and self.current_patience >= self.patience:
                print("Stopping early at epoch {}!".format(epoch+1))
                break

        # At the end of training swap the best params into the model
        self.model.parameters = self.best_params

        print('FINISH.')


    def update_best_loss(self, val_loss, train_loss):
        # Update the model and best loss if we see improvements.
        if not self.best_model_stats or val_loss < self.best_model_stats["val_loss"]:
            self.best_model_stats = {"val_loss":val_loss, "train_loss":train_loss}
            self.best_params = self.model.parameters
            self.current_patience = 0
        else:
            self.current_patience += 1
