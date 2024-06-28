# %%
import json, random, multiprocessing
from tqdm import tqdm
from datasets import Dataset
import torchaudio

# %%

def load_json(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

pair_list = load_json('./aishell3_list.json') + load_json('./thchs30_list.json')
random.shuffle(pair_list)


# %%
def process_pair(pair):
    audio_file = pair[0]
    trans = pair[1]
    text = ' '.join(pair[2])

    audio, sr = torchaudio.load(audio_file)
    resampled_audio = torchaudio.functional.resample(audio, sr, 16000)

    return audio_file, trans, resampled_audio[0], text

pair_dict = {'audio': [], 'text': [], 'path': [], 'trans': []}

for p in tqdm(pair_list):
    audio_file, trans, audio_data, text = process_pair(p)
    pair_dict['audio'].append({'array': audio_data})
    pair_dict['text'].append(text)
    pair_dict['path'].append(audio_file)
    pair_dict['trans'].append(trans)

ds = Dataset.from_dict(pair_dict)
ds = ds.train_test_split(test_size=0.1)

ds.save_to_disk('../ds4')


# %%



