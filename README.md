# KoSimCSE
Korean Simple Contrastive Learning of Sentence Embeddings <br>

## Installation
```
git clone https://github.com/BM-K/KoSimCSE.git
pip install -r requirements.txt
pip install .
```
## Training - only supervised
 - Model
    - [SKT KoBERT](https://github.com/SKTBrain/KoBERT)
 - Dataset
    - [kakaobrain NLU dataset](https://github.com/kakaobrain/KorNLUDatasets)
      - train: KorNLI
      - dev & test: KorSTS
 - Run training
  ```
  bash run_example.sh
  ```
## Pre-Trained Models
  - BERT [CLS] token representation 사용하여 학습.
  - Pre-Trained model check point <br>
    - [Google Sharing]()

## Performance
|Model|Cosine Pearson|Cosine Spearman|Euclidean Pearson|Euclidean Spearman|Manhattan Pearson|Manhattan Spearman|Dot Pearson|Dot Spearman|
|:------------------------:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|KoSBERT_SKT*|78.81|78.47|77.68|77.78|77.71|77.83|75.75|75.22|
|KoSimCSE_SKT|0|0|0|0|0|0|0|0|

