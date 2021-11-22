'''
this shit will be called to train the attention part

steps that it will be performing:
1. define hyperparams, like batch_size, learning rate, optimizer, etc. We can also pass them while calling .py file
2. split into train-val and create data generator objects (this shit will be tricky because attention model is receiving different data shape than CNN part.
this needs to be handled either in models.py or in data_generator.py. I guess models.py is better)
3. create model object, compile it and train by using keras' fit
4. save the model somewhere in repo (we will need to add it .gitignore because its gonna be huge probably)
'''