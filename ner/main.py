import torch
from data import getData_FSC
from train import train, test
from model import Seq2SeqBiLSTM
import matplotlib.pyplot as plt
import argparse
from transformers import Wav2Vec2ForCTC

parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')

parser.add_argument('--test', type=bool, help='Indicate Test')

args = parser.parse_args()

# Define your Seq2SeqBiLSTM model
embedding_dim = 768
hidden_dim = 256
output_dim = 3
num_layers = 2

# Training loop
epochs = 20  # Update this to the number of training epochs

# Example usage
seq_length = 15  # Update this to your desired sequence length
batch_size = 16  # Update this to your desired batch size

device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

model_name = 'vasista22/ccc-wav2vec2-base-100h'
# model_name = 'facebook/wav2vec2-base-960h'

embed_model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)

layer = embed_model.wav2vec2.encoder.layers[11].final_layer_norm

model = Seq2SeqBiLSTM(embedding_dim, hidden_dim, output_dim, embed_model, layer, num_layers, device)
model.to(device)

if not args.test:
    train_loader = getData_FSC('train',batch_size)
    trained_model, loss_history = train(model,train_loader, epochs, device)
    torch.save(trained_model.state_dict(),'./saved_models/e2e_ner_lstm.pt')
    plt.plot(loss_history)
    plt.xticks([i for i in range(epochs)])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('./training_curve.png')


model.load_state_dict(torch.load('./saved_models/e2e_ner_lstm.pt'))
valid_loader = getData_FSC('valid',batch_size)
test(model, valid_loader)
test_loader = getData_FSC('test',batch_size)
test(model, test_loader)
