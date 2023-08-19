import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import torchaudio

dataset_path_base = '../../dataset/fluent_speech_commands_dataset/'


def train(model, dataloader, epochs=10, device='cpu'):
    # Define your loss function and optimizer
    labels_fa = (torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H).get_labels()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001,eps=1e-08,weight_decay=0.01)
    loss_history = []
    for epoch in tqdm(range(epochs)):
        running_loss = 0
        for batch in tqdm(dataloader):
            optimizer.zero_grad()
            target = torch.stack(batch['entities']).view(-1).to(device)
            output_sequence = model(batch, labels_fa)
            a,b,c = output_sequence.shape
            # Calculate loss
            loss = criterion(output_sequence.reshape(a*b, 3), target)
            running_loss += loss.detach().cpu().item()
            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
        running_loss = running_loss
        loss_history.append((running_loss/len(dataloader)))
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    return model,loss_history

def test(model, dataloader):
    labels_fa = (torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H).get_labels()
    acc = 0
    for batch in tqdm(dataloader):
        target = torch.stack(batch['entities'])
        with torch.no_grad():
            output_sequence = model(batch,labels_fa)
        output_sequence = output_sequence.detach().cpu()
        op = torch.argmax(output_sequence.permute(0,2,1), dim = 1)
        acc += torch.sum(op[:,:5]==target[:,:5])/(len(op)*5)
    print(f'Accuracy: {(acc/len(dataloader)):0.4f}')