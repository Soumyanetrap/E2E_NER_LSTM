import torch
import torch.nn as nn
from forced_allign import *
import torch.nn.functional as F
import re

dataset_path_base = '../../dataset/fluent_speech_commands_dataset'


def hook(module, input, output):
    global layer_output
    layer_output = output

class Seq2SeqBiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, embed_model, layer, num_layers=1, device='cpu'):
        super(Seq2SeqBiLSTM, self).__init__()
        torch.manual_seed(10)
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device
        self.embed_model = embed_model
        self.layer = layer

        # Encoder
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        for name, param in self.encoder.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        # Decoder
        # self.decoder = nn.LSTM(hidden_dim*2, hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Sequential(
                                nn.Linear(2*hidden_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim//2),
                                nn.ReLU(),
                                nn.Linear(hidden_dim//2, hidden_dim//4),
                                nn.ReLU(),
                                nn.Linear(hidden_dim//4, output_dim),
                            )
        for layer in self.output_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def embedding(self, batch, labels):
        maxlen = 15
        input_seq = []

        hook_handle = self.layer.register_forward_hook(hook)
        # retrieve logits
        _= self.embed_model(batch['input_values'].to(self.device))

        batch_out = layer_output
        hook_handle.remove()
        for out,transcript in zip(batch_out,batch['transcripts']):
            # take argmax and decode    
            chars_to_ignore_regex = '[\,\?\.\-\;\:\’]'
            transcript = re.sub(chars_to_ignore_regex, '', transcript)
            transcript = re.sub("\s", "|",transcript).upper()
            
            dictionary = {c: i for i, c in enumerate(labels)}
            tokens = [dictionary[c] for c in transcript]
            trellis = get_trellis(out, tokens, device=self.device)
            transition_path = backtrack(trellis, out, tokens)
            segments = merge_repeats(transition_path, transcript)
            word_segments = merge_words(segments)
            word_embeds = []
            for word in word_segments:
                word_embeds.append(torch.mean(F.normalize(out[word.start:word.end]),0,True))
            for _ in range(maxlen-len(word_embeds)):
                word_embeds.append(torch.zeros(1,768).to(self.device))
            input_seq.append(torch.stack(word_embeds))
        return torch.stack(input_seq).squeeze(2)
    
    def forward(self, batch, labels):
        #Embedding
        input_seq = self.embedding(batch, labels).to(self.device)
        
        
        # Encoder
        encoder_output, (encoder_hidden, encoder_cell) = self.encoder(input_seq)
        
        # # Initialize decoder hidden states and cells with the final encoder states
        # decoder_hidden = encoder_hidden[:len(encoder_hidden)//2,:,:]
        # decoder_cell = encoder_cell[:len(encoder_cell)//2,:,:]
        # # Decoder
        decoder_outputs = []
        # decoder_output, (decoder_hidden, decoder_cell) = self.decoder(encoder_output, (decoder_hidden, decoder_cell))
        for t in range(input_seq.size(1)):
            linear_output = self.output_layer(encoder_output[:,t,:])
            decoder_outputs.append(linear_output)
        
        decoder_outputs = torch.stack(decoder_outputs).permute(1,0,2)
        # print(decoder_outputs.shape)
        return decoder_outputs
