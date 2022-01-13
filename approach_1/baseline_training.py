import torch

class BaselineSolver(object):

    def __init__(self, model, train_dataloader, val_dataloader):

        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.batch_size = model.hparams['batch_size']
        self.learning_rate = model.hparams['learning_rate']
        self.lr_decay = model.hparams['lr_decay']
        self.epochs = model.hparams['epochs']
        self.loss_func = model.hparams['loss_func']
        self.optimizer = model.hparams['optimizer'](list(self.model.parameters()), lr=self.learning_rate)

        #self.current_patience = 0
        #self._reset()


    def _step(self, images, labels):
        loss = None
        self.optimizer.zero_grad()
        predictions = self.model.forward(torch.tensor(images, dtype=torch.float32))
        loss = self.loss_func(predictions, labels)
        loss.backward()
        self.optimizer.step()
        return loss


    def train(self):

        train_loss_history = [] # loss
        train_acc_history = [] # accuracy

        for epoch in range(self.epochs):
            
            running_loss = 0.0
        
            # Iterating through the minibatches of the data
            for i, batch in enumerate(self.train_dataloader):
                
                images, labels = batch['image'], batch['label']

                labels = torch.tensor(labels, dtype=torch.long)

                #image = image.to(device)
                #labels = labels.to(device)
        
                train_loss = self._step(images, labels)
                running_loss += train_loss.item()

                # Print statistics to console
                if i % 1 == 0: # print every 10 mini-batches
                    running_loss /= 1
                    print("[Epoch %d, Iteration %5d] loss: %.5f" % (epoch+1, i+1, running_loss))
                    train_loss_history.append(running_loss)
                    running_loss = 0.0

        print('FINISH.')


