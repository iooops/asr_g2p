# %%
# %env CUDA_VISIBLE_DEVICES=8

# %%
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_from_disk, load_metric
import numpy as np

# %%
ds = load_from_disk('./ds4')
# ds = ds.train_test_split(test_size=0.01)

# %%
from datasets import ClassLabel
import random
import pandas as pd
from IPython.display import display, HTML

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    display(HTML(df.to_html()))

show_random_elements(ds["train"].remove_columns(["audio"]))


# %%
def extract_all_chars(batch):
  all_text = " ".join(batch["text"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}

vocabs = ds.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=ds.column_names["train"])

# %%
tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# %%
def prepare_dataset(batch):
    resampled_audio = batch["audio"]["array"]
    batch["input_values"] = processor(resampled_audio, sampling_rate=16000).input_values[0]
    with processor.as_target_processor():
        batch["labels"] = processor(batch['text']).input_ids
    return batch

# %%
ds = ds.map(prepare_dataset, remove_columns=ds.column_names["train"], num_proc=32)

# %%
import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

# %%
cer_metric = load_metric("cer")

# %%
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}


class ExtendedWav2Vec2ForCTC(Wav2Vec2ForCTC):
    """
    In ESPNET there is a LayerNorm layer between encoder output and CTC classification head.
    """
    def __init__(self, config):
        super().__init__(config)
        self.lm_head = torch.nn.Sequential(
                torch.nn.LayerNorm(config.hidden_size),
                self.lm_head
        )

# %%
model = ExtendedWav2Vec2ForCTC.from_pretrained(
    # "facebook/wav2vec2-large-xlsr-53", 
    # "facebook/wav2vec2-base",
    # "kehanlu/mandarin-wav2vec2-aishell1",
    "TencentGameMate/chinese-wav2vec2-base",
    # "TencentGameMate/chinese-wav2vec2-large",
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer)
)

model.to('cuda')

# %%
model.freeze_feature_extractor()

# %%
from transformers import TrainingArguments

training_args = TrainingArguments(
  output_dir='trained_full_tgml_n_fixed_t_base_ds4',
  # group_by_length=True,
  per_device_train_batch_size=8,
  evaluation_strategy="steps",
  num_train_epochs=30,
  fp16=True,
  gradient_checkpointing=True, 
  save_steps=500,
  eval_steps=500,
  logging_steps=500,
  learning_rate=1e-4,
  weight_decay=0.005,
  warmup_steps=1000,
  save_total_limit=50,
  eval_accumulation_steps=8
)

# %%
from transformers import Trainer

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    tokenizer=processor.feature_extractor,
)


# %%
trainer.train()

# %%



