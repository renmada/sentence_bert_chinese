# Pretrain [sentence transformers](https://github.com/UKPLab/sentence-transformers) on Chinese text matching dataset. 

## Traing Data
Inlcuding **STS**(Semantic Textual Similarity) task and **NLI** (Natural Language Inference) task.  
Most datas are from [CLUEDatasetSearch](https://github.com/CLUEbenchmark/CLUEDatasetSearch/tree/master/%E6%96%87%E6%9C%AC%E5%8C%B9%E9%85%8D).

## Training Detail
Acording to the paper, after training 1 epoch on NLI data, training 2 epoches on STS data.  
The original BERT from [ymcui/Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm), using **RTB3**(small size) and **Robert_wwm_ext**(bert_base size)  

# Getting Model

## use Huggingface-Transformers

|  model   | model_name  |
|  ----  | ----  |
| rtb3  | imxly/sentence_rtb3 |
| roberta_wwm_ext  | imxly/sentence_roberta_wwm_ext |

# How to use


```
pip install sentence_transformers
```

```
from sentence_transformers import models, SentenceTransformer

model_name = 'imxly/sentence_rtb3'
word_embedding_model = models.Transformer(model_name)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


def evaluate(model):
    '''
    余弦相似度计算
    '''
    import numpy as np
    v1 = model.encode(s1)
    v2 = model.encode(s2)
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    return v1.dot(v2)


s1 = '公积金贷款能贷多久'
s2 = '公积金贷款的期限'
print(evaluate(model))```
