import torch
import pandas as pd
import os
import librosa
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor

dataset_path_base = '../../dataset/fluent_speech_commands_dataset'
model_name = 'vasista22/ccc-wav2vec2-base-100h'
processor = Wav2Vec2Processor.from_pretrained(model_name)

tag_dict = {
    'O': 0, 'object': 1, 'location': 2
}

class Dataset:
    def __init__(self,dataframe):
        self.dataframe = dataframe
    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self, idx):
        path = self.dataframe.loc[idx, 'path']
        transcript = self.dataframe.loc[idx, 'transcription']
        entity = self.dataframe.loc[idx,'labels']

        return{
            'path':path,
            'transcript': transcript,
            'entity': entity
        }


def collate_data(batch):
    maxlen = 15
    paths = [item['path'] for item in batch]
    batch_audio = []
    for filePath in paths:
        audio,rate = librosa.load(os.path.join(dataset_path_base,filePath), sr=16000)
        batch_audio.append(audio)
    
    input_values = processor(batch_audio, sampling_rate=rate, return_tensors="pt", padding="longest").input_values
    transcripts = [item['transcript'] for item in batch]
    entities = [item['entity'].split(',') for item in batch]
    for i in range(len(entities)):
        entities[i] = torch.tensor([*[tag_dict[item] for item in entities[i]],*[tag_dict['O'] for _ in range(maxlen-len(entities[i]))]])
        
    return {
            'input_values': input_values,
            'transcripts' : transcripts,
            'entities': entities
    }


def getData_FSC(type, batch_size):
    if type == 'train':
        train_df = pd.read_csv(os.path.join(dataset_path_base+'/data','train_labeled.csv'))
        train_data = Dataset(train_df)
        dataloader = DataLoader(train_data,batch_size=batch_size, shuffle=True, collate_fn=collate_data)

    elif type == 'valid':
        valid_df = pd.read_csv(os.path.join(dataset_path_base+'/data','valid_labeled.csv'))
        valid_data = Dataset(valid_df)
        dataloader = DataLoader(valid_data,batch_size=batch_size, shuffle=True, collate_fn=collate_data)
    else:

        test_df = pd.read_csv(os.path.join(dataset_path_base+'/data','test_labeled.csv'))
        test_data = Dataset(test_df)
        dataloader = DataLoader(test_data,batch_size=batch_size, shuffle=True, collate_fn=collate_data)

    return dataloader

# for item in dataloader:
#     print(len(item['path']))
#     break
