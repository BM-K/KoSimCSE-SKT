# KoSimCSE
 - Korean Simple Contrastive Learning of Sentence Embeddings Implementation <br>
   - [SimCSE](https://arxiv.org/abs/2104.08821) <br> <br>
 <img src=https://user-images.githubusercontent.com/55969260/128805705-2381ad67-edc2-4070-ae36-a266122b9319.png> <br>
## Installation
```
git clone https://github.com/BM-K/KoSimCSE.git
cd KoSimCSE
git clone https://github.com/SKTBrain/KoBERT.git
cd KoBERT
pip install -r requirements.txt
pip install .
cd ..
pip install -r requirements.txt
```
## Training - only supervised
 - Model
    - [SKT KoBERT](https://github.com/SKTBrain/KoBERT)
    
 - Dataset
    - [kakaobrain NLU dataset](https://github.com/kakaobrain/KorNLUDatasets)
      - train: KorNLI
      - dev & test: KorSTS
      
 - Setting
   - epochs: 3
   - dropout: 0.1
   - batch size: 256
   - temperature: 0.05
   - learning rate: 1e-4
   - warm-up ratio: 0.05
   - max sequence length: 50
   - evaluation steps during training: 250
   
 - Run train -> test -> semantic_search
  ```
  bash run_example.sh
  ```
## Pre-Trained Models
  - Using BERT pooled [CLS] token representation
    - It may be better to use only the [CLS] token representation, not pooled
  - Pre-Trained model check point <br>
    - [Google Drive Sharing](https://drive.google.com/drive/folders/1qiqqIucgqavAMmAn1HFJyLL9LZ2U6cbx?usp=sharing)
    - ./output/nli_checkpoint.pt

## Performance 
| Model                  | AVG | Cosine Pearson | Cosine Spearman | Euclidean Pearson | Euclidean Spearman | Manhattan Pearson | Manhattan Spearman | Dot Pearson | Dot Spearman |
|------------------------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| KoSBERT<sup>†</sup><sub>SKT</sub>    | 77.40 | 78.81 | 78.47 | 77.68 | 77.78 | 77.71 | 77.83 | 75.75 | 75.22 |
| KoSBERT              | 80.39 | 82.13 | 82.25 | 80.67 | 80.75 | 80.69 | 80.78 | 77.96 | 77.90 |
| KoSRoBERTa           | 81.64 | 81.20 | 82.20 | 81.79 | 82.34 | 81.59 | 82.20 | 80.62 | 81.25 |
| | | | | | | | | |
| KoSentenceBART         | 77.14 | 79.71 | 78.74 | 78.42 | 78.02 | 78.40 | 78.00 | 74.24 | 72.15 |
| KoSentenceT5          | 77.83 | 80.87 | 79.74 | 80.24 | 79.36 | 80.19 | 79.27 | 72.81 | 70.17 |
| | | | | | | | | |
| **KoSimCSE-BERT<sup>†</sup><sub>SKT</sub>**   | 81.32 | 82.12 | 82.56 | 81.84 | 81.63 | 81.99 | 81.74 | 79.55 | 79.19 |
| KoSimCSE-BERT              | 83.37 | 83.22 | 83.58 | 83.24 | 83.60 | 83.15 | 83.54 | 83.13 | 83.49 |
| KoSimCSE-RoBERTa          | 83.65 | 83.60 | 83.77 | 83.54 | 83.76 | 83.55 | 83.77 | 83.55 | 83.64 |
| | | | | | | | | | |
| KoSimCSE-BERT-multitask              | 85.71 | 85.29 | 86.02 | 85.63 | 86.01 | 85.57 | 85.97 | 85.26 | 85.93 |
| KoSimCSE-RoBERTa-multitask          | 85.77 | 85.08 | 86.12 | 85.84 | 86.12 | 85.83 | 86.12 | 85.03 | 85.99 |

 - †: [KoSBERT<sup>†</sup><sub>SKT</sub>](https://github.com/BM-K/KoSentenceBERT_SKT)
 - Performance comparison with other models [[KLUE-PLMs]](https://github.com/BM-K/Sentence-Embedding-is-all-you-need#korean-sentence-embedding).
 
## Example Downstream Task
### Semantic Search
```
python SemanticSearch.py
```
```python
import numpy as np
from model.utils import pytorch_cos_sim
from data.dataloader import convert_to_tensor, example_model_setting


def main():
    model_ckpt = './output/nli_checkpoint.pt'
    model, transform, device = example_model_setting(model_ckpt)

    # Corpus with example sentences
    corpus = ['한 남자가 음식을 먹는다.',
              '한 남자가 빵 한 조각을 먹는다.',
              '그 여자가 아이를 돌본다.',
              '한 남자가 말을 탄다.',
              '한 여자가 바이올린을 연주한다.',
              '두 남자가 수레를 숲 속으로 밀었다.',
              '한 남자가 담으로 싸인 땅에서 백마를 타고 있다.',
              '원숭이 한 마리가 드럼을 연주한다.',
              '치타 한 마리가 먹이 뒤에서 달리고 있다.']

    inputs_corpus = convert_to_tensor(corpus, transform)

    corpus_embeddings = model.encode(inputs_corpus, device)

    # Query sentences:
    queries = ['한 남자가 파스타를 먹는다.',
               '고릴라 의상을 입은 누군가가 드럼을 연주하고 있다.',
               '치타가 들판을 가로 질러 먹이를 쫓는다.']

    # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
    top_k = 5
    for query in queries:
        query_embedding = model.encode(convert_to_tensor([query], transform), device)
        cos_scores = pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
        cos_scores = cos_scores.cpu().detach().numpy()

        top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]

        print("\n\n======================\n\n")
        print("Query:", query)
        print("\nTop 5 most similar sentences in corpus:")

        for idx in top_results[0:top_k]:
            print(corpus[idx].strip(), "(Score: %.4f)" % (cos_scores[idx]))
```
### Result
```
Query: 한 남자가 파스타를 먹는다.

Top 5 most similar sentences in corpus:
한 남자가 음식을 먹는다. (Score: 0.6146)
한 남자가 빵 한 조각을 먹는다. (Score: 0.4922)
한 남자가 말을 탄다. (Score: 0.0797)
한 남자가 담으로 싸인 땅에서 백마를 타고 있다. (Score: 0.0183)
한 여자가 바이올린을 연주한다. (Score: 0.0041)


======================


Query: 고릴라 의상을 입은 누군가가 드럼을 연주하고 있다.

Top 5 most similar sentences in corpus:
원숭이 한 마리가 드럼을 연주한다. (Score: 0.5087)
한 여자가 바이올린을 연주한다. (Score: 0.4180)
한 남자가 말을 탄다. (Score: 0.3403)
그 여자가 아이를 돌본다. (Score: 0.2689)
한 남자가 담으로 싸인 땅에서 백마를 타고 있다. (Score: 0.1671)


======================


Query: 치타가 들판을 가로 질러 먹이를 쫓는다.

Top 5 most similar sentences in corpus:
치타 한 마리가 먹이 뒤에서 달리고 있다. (Score: 0.8106)
한 남자가 말을 탄다. (Score: 0.1910)
한 남자가 담으로 싸인 땅에서 백마를 타고 있다. (Score: 0.1614)
두 남자가 수레를 숲 속으로 밀었다. (Score: 0.1557)
원숭이 한 마리가 드럼을 연주한다. (Score: 0.1269)


```

## Citing
```bibtex
@article{gao2021simcse,
   title={{SimCSE}: Simple Contrastive Learning of Sentence Embeddings},
   author={Gao, Tianyu and Yao, Xingcheng and Chen, Danqi},
   journal={arXiv preprint arXiv:2104.08821},
   year={2021}
}
@article{ham2020kornli,
 title={KorNLI and KorSTS: New Benchmark Datasets for Korean Natural Language Understanding},
 author={Ham, Jiyeon and Choe, Yo Joong and Park, Kyubyong and Choi, Ilji and Soh, Hyungjoon},
 journal={arXiv preprint arXiv:2004.03289},
 year={2020}
}
```
