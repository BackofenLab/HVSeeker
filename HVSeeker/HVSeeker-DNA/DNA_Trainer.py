import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils import class_weight as clw
import numpy as np
import matplotlib.pyplot as plt
import shutil
import sys

os.environ["TORCH_USE_CUDA_DSA"] = "1"

class DNALSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(DNALSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, bidirectional=True, dropout=dropout, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim*2, hidden_dim, bidirectional=True, dropout=dropout, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_dim*2, hidden_dim, dropout=dropout, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
    


        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        x = self.elu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)


        return x

def DNA_model(X_train, X_val, Y_train, Y_val, outpath, sampleSize=1, nodes=32, suffix="", epochs=100, dropout=0, shuffleTraining=True):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 64
    learning_rate = 0.001
    Y_train_noOHE = [y.argmax() for y in Y_train]
    class_weight = clw.compute_class_weight(class_weight = 'balanced', classes = np.unique(Y_train_noOHE), y = Y_train_noOHE)
    class_weight_dict = {i: class_weight[i] for i in range(len(class_weight))}
    class_weight = torch.FloatTensor(list(class_weight_dict.values())).to(device)

    timesteps = X_val.shape[1]

    model = DNALSTM(input_dim=X_val.shape[-1], hidden_dim=nodes, output_dim=Y_train.shape[-1], dropout=dropout).to(device)


    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs) 


    train_dataset = TensorDataset(torch.tensor(X_train[0:int(len(X_train) / sampleSize)], dtype=torch.float32),
                                  torch.tensor(Y_train_noOHE[0:int(len(Y_train) / sampleSize)], dtype=torch.long))


                                  
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                torch.tensor([y.argmax() for y in Y_val], dtype=torch.long))


                                
                                

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffleTraining)
    
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = float('inf')
    best_val_acc = 0
    train_losses = []
    val_losses = []
    val_accuracies = []
    train_accuracies = []
    
    epoch = 0
    log_num = 0
    
    
    if os.path.isfile(outpath + "/model_dict_save.pt"):
    
    
        model_dictionary = torch.load(outpath + "/model_dict_save.pt")
        log_num = model_dictionary["log_num"] + 1
        
        if log_num > 100:
            sys.exit(0)
        
        
        epoch = model_dictionary["epoch"]
        best_val_loss =  model_dictionary["best_val_loss"]
        best_val_acc =  model_dictionary["best_val_acc"]
        model.load_state_dict(model_dictionary["model_dict"])
        optimizer.load_state_dict(model_dictionary["optimizer_dict"])
        scheduler.load_state_dict(model_dictionary["scheduler_dict"])
        shutil.copyfile(outpath + "/model_dict_save.pt", outpath + "/model_dict_save" + "_iteration_"+ str(model_dictionary["log_num"]) + ".pt")
        
        
        
    while epoch < epochs:
    
    
    #for epoch in range(epochs):
        # torch.cuda.empty_cache()
        
        
        epoch_train_loss = 0
        num_batches = 0
        correct_train = 0  # Count of correct predictions during training
        total_train = 0  # Total count of training samples
        model.train()
        for batch, (x, y) in enumerate(train_loader):
            # torch.cuda.empty_cache()
            
            x, y = x.to(device), y.to(device)
            # print(f"Batch: {batch}, x shape: {x.shape}, y shape: {y.shape}")
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            # scheduler.step()

            epoch_train_loss += loss.item()  # Accumulate batch loss
            num_batches += 1
            _, predicted_train = torch.max(output.data, 1)
            total_train += y.size(0)
            correct_train += (predicted_train == y).sum().item()
            
        train_loss = epoch_train_loss / num_batches
        train_losses.append(train_loss)

        train_acc = 100 * correct_train / total_train  # Calculate training accuracy
        train_accuracies.append(train_acc)
        scheduler.step()

                
        
        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        with torch.no_grad():
            for batch, (x, y) in enumerate(val_loader):
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = criterion(output, y)
                _, predicted = torch.max(output.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                val_loss += loss.item()
               

        val_loss /= len(val_loader)
        val_acc = 100 * correct / total

        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        epoch = epoch + 1
        
        
        torch.save({"model_dict": model.state_dict(), "optimizer_dict": optimizer.state_dict(), "scheduler_dict": scheduler.state_dict(), "epoch": epoch, "best_val_loss": best_val_loss, "best_val_acc": best_val_acc, "log_num": log_num}, outpath + "/model_dict_save.pt")
        
        
        
        with open('log' + str(log_num) + '.txt', 'a') as f:
            print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss}, Train Acc: {train_acc}, Val Loss: {val_loss}, Val Acc: {val_acc}', file=f)

        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss}, Train Acc: {train_acc}, Val Loss: {val_loss}, Val Acc: {val_acc}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, outpath + "/model_best_loss2_" + suffix + ".pt")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model, outpath + "/model_best_acc2_" + suffix + ".pt")


    plot_metrics(train_losses, train_accuracies, val_losses, val_accuracies)

    return model


def plot_metrics(train_losses, train_accuracies, val_losses, val_accuracies):
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss.png')  # Save the figure
    plt.show()

    # Plot validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig('accuracy.png')  # Save the figure
    plt.show()
