
import torch
from torch.utils.data import DataLoader,TensorDataset

class TrainingHelper:
    def __init__(self,n_history,train_set_scaled,val_set_scaled,test_set_scaled):
        self.n_history = n_history
        self.train_set_scaled = train_set_scaled
        self.val_set_scaled = val_set_scaled
        self.test_set_scaled = test_set_scaled


    def generate_tensors(self,hidden_size):

        X_train = torch.empty((len(self.train_set_scaled)-self.n_history, self.n_history, hidden_size))


        y_train = torch.empty(len(self.train_set_scaled)-self.n_history)

        X_val = torch.empty((len(self.val_set_scaled)-self.n_history, self.n_history, hidden_size))
        y_val = torch.empty(len(self.val_set_scaled)-self.n_history)


        X_test = torch.empty((len(self.test_set_scaled)-self.n_history, self.n_history, hidden_size))
        y_test = torch.empty(len(self.test_set_scaled)-self.n_history)

        for k in range(0,len(self.train_set_scaled)-self.n_history):
            X_train[k] = torch.from_numpy(self.train_set_scaled[k:k+self.n_history].values)
            y_train[k] = self.train_set_scaled.iloc[k+self.n_history].Open

        for k in range(0,len(self.val_set_scaled)-self.n_history):
                X_val[k] = torch.from_numpy(self.val_set_scaled[k:k+self.n_history].values)
                y_val[k] = self.val_set_scaled.iloc[k+self.n_history].Open
                
        for k in range(0,len(self.test_set_scaled)-self.n_history):
                X_test[k] = torch.from_numpy(self.test_set_scaled[k:k+self.n_history].values)
                y_test[k] = self.test_set_scaled.iloc[k+self.n_history].Open

        return {
            "train": (X_train,y_train),
            "val": (X_val,y_val),
            "test": (X_test,y_test)
        }

    def generate_dataloader(self,batch_size,hidden_size):
        X_train,y_train = self.generate_tensors(hidden_size)['train']
        X_val,y_val = self.generate_tensors(hidden_size)['val']
        X_test,y_test = self.generate_tensors(hidden_size)['test']

        train = TensorDataset(X_train, y_train)
        val = TensorDataset(X_val, y_val)
        test = TensorDataset(X_test, y_test)

        
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
        val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)
        test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
        test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)

        return train_loader,val_loader,test_loader,test_loader_one
    
