Pretrain [sentence transformers](https://github.com/UKPLab/sentence-transformers) on Chinese text matching dataset while inlcuding **STS**(Semantic Textual Similarity) task and **NLI** (Natural Language Inference) task.  
Most Data are from [CLUEDatasetSearch].(https://github.com/CLUEbenchmark/CLUEDatasetSearch/tree/master/%E6%96%87%E6%9C%AC%E5%8C%B9%E9%85%8D)  
Acording to the paper, after training 1 epoch on NLI data, training 2 epoches on STS data.  
The original BERT from [ymcui/Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm), using **RTB3**(small_size) and **Robert_wwm_ext**(bert_base_size)  

# Getting Model

## use Huggingface-Transformers

|  model   | model_name  |
|  ----  | ----  |
| rtb3  | sentence_rtb3 |
| roberta_wwm_ext  | sentence_roberta_wwm_ext |