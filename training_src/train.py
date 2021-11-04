import os
from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer, util, InputExample
import logging
from datetime import datetime
import sys
import glob
from module import Transformer
import pandas as pd


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

nli_dataset_path = './data/nli/nlis.csv'
sts_dataset_path = glob.glob('./data/sts/*.csv')

model_name = sys.argv[1]
train_batch_size = 96


model_save_path = 'output/training_nli_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
# word_embedding_model = models.Transformer(model_name)
word_embedding_model = Transformer(model_name)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


# Read the AllNLI.tsv.gz file and create the training dataset
logging.info("Read AllNLI train dataset")

label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
train_samples = []
nli = pd.read_csv(nli_dataset_path).dropna()
nli.label = nli.label.astype('int64')
# print(nli.info())
for row in nli.values:
    train_samples.append(InputExample(texts=[row[0], row[1]], label=row[2]))

train_dataset = SentencesDataset(train_samples, model=model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=len(label2int))

logging.info("Read STSbenchmark dev dataset")

dev_samples = []
for i in sts_dataset_path:
    df = pd.read_csv(i)
    df.iloc[:, -1] = df.iloc[:, -1].astype('float32')
    for row in df.values:
        dev_samples.append(InputExample(texts=[row[0], row[1]], label=row[2]))
print('num of dev_samples', len(dev_samples))

sts_dataset = SentencesDataset(dev_samples, model=model)
sts_dataloader = DataLoader(sts_dataset, shuffle=True, batch_size=train_batch_size)
sts_loss = losses.CosineSimilarityLoss(model=model)

num_epochs = 1

warmup_steps = math.ceil(len(train_dataset) * num_epochs / train_batch_size * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=1,
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          use_amp=True
          )

model.fit(train_objectives=[(sts_dataloader, sts_loss)],
          epochs=2,
          output_path=model_save_path,
          use_amp=True
          )

model.save('albert_small_sbert')
