import torch
from tqdm import tqdm
from transformers import get_scheduler
from datasets import load_dataset, load_metric
from transformers import CanineTokenizer, CanineForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification

from textattack.attack_recipes.pruthi_2019 import Pruthi2019
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import ModelWrapper
from transformers import pipeline
from textattack import Attacker

import numpy as np
import random

# Set Random Seeds
torch.backends.cudnn.deterministic = True
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

# Load Dataset
dataset = load_dataset("imdb")

def tokenize_function(tokenizer, input_field):
    return lambda examples: tokenizer(examples[input_field], padding="max_length", truncation=True)

# Load and test CANINE
canine_tokenizer = CanineTokenizer.from_pretrained("google/canine-s")
canine = CanineForSequenceClassification.from_pretrained("google/canine-s", num_labels=2)

inputs = canine_tokenizer("Hello, my dog is cute", return_tensors="pt")
labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
outputs = canine(**inputs, labels=labels)
print(outputs)
loss = outputs.loss
logits = outputs.logits

inputs = canine_tokenizer("Hello, my dog is cute", return_tensors="pt")



# Load and test BERT
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

inputs = bert_tokenizer("Hello, my dog is cute", return_tensors="pt")
labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
outputs = bert(**inputs, labels=labels)
print(outputs)
loss = outputs.loss
logits = outputs.logits



# Training the models 

# Select model
model = canine #bert
tokenizer = canine_tokenizer #bert_tokenizer

# Parameters
learning_rate = 1e-5
n_epochs = 5
batch_size = 4
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Dataset parameter
dataset_input_field = "text"
dataset_label_field = "label"
#dataset_remove_columns = ['count','hate_speech_count', 'offensive_language_count', 'neither_count', 'tweet']
# Tokenizer wraggling
tokenized_datasets = dataset.map(tokenize_function(tokenizer, dataset_input_field), batched=True)
#tokenized_datasets = tokenized_datasets.remove_columns(dataset_remove_columns)
tokenized_datasets = tokenized_datasets.remove_columns(['text'])
tokenized_datasets = tokenized_datasets.rename_column(dataset_label_field, "labels")
tokenized_datasets.set_format("torch")

#tokenized_datasets = tokenized_datasets['train'].train_test_split(0.1) # Remove if already split into train/test

# Loaders
train_dataloader = torch.utils.data.DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=batch_size)
test_dataloader = torch.utils.data.DataLoader(tokenized_datasets["test"], batch_size=batch_size)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
n_training_steps = n_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=n_training_steps)
metric = load_metric("accuracy")

print("Features:")
print(tokenized_datasets['train'].features)

for k in range(n_epochs):
    # Train
    model.train()
    for batch in tqdm(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    # Test
    model.eval()
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    metric.compute()

torch.cuda.empty_cache()

# Attacks

class PipelineWrapper(ModelWrapper):
    """ Transformers sentiment analysis pipeline returns a list of responses
        like

            [{'label': 'POSITIVE', 'score': 0.7817379832267761}]

        We need to convert that to a format TextAttack understands, like

            [[0.218262017, 0.7817379832267761]]
    """
    def __init__(self, model):
        self.model = model#pipeline = pipeline
        #self.softmax = torch.nn.Softmax(dim=1)
    def __call__(self, text_inputs):
        raw_outputs = self.model(text_inputs)
        outputs = []
        for output in raw_outputs:
            score = output['score']
            if output['label'] == 'LABEL_1':
                outputs.append([1-score, score])
            else:
                outputs.append([score, 1-score])
        return np.array(outputs)

# Shuffle test split dataset

test_dataset = dataset['test'].shuffle(seed=42)

# Prepare attack 
model_wrapper = PipelineWrapper(pipeline)
recipe = Pruthi2019.build(model_wrapper)

list_attacks = []

for i in range (0, len(test_dataset),10):
    small_dataset = HuggingFaceDataset(test_dataset.select(list(range(i, min(i+10, len(test_dataset))))))
    attacker = Attacker(recipe, dataset1)
    results = attacker.attack_dataset()
    list_attacks.append(results)

print(list_attacks)