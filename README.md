# KoSimCSE
 - Korean Simple Contrastive Learning of Sentence Embeddings implementation using pytorch<br>
   - [SimCSE](https://arxiv.org/abs/2104.08821)
 
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
      
 - Setting
   - epochs: 3
   - dropout: 0.1
   - batch size: 256
   - temperature: 0.05
   - learning rate: 5e-5
   - warm-up ratio: 0.05
   - max sequence length: 50
   - evaluation steps during training: 250
   
 - Run training
  ```
  bash run_example.sh
  ```
## Pre-Trained Models
  - Using BERT [CLS] token representation
  - Pre-Trained model check point <br>
    - [Google Sharing](https://drive.google.com/drive/folders/1qiqqIucgqavAMmAn1HFJyLL9LZ2U6cbx?usp=sharing)

## Performance
|Model|Cosine Pearson|Cosine Spearman|Euclidean Pearson|Euclidean Spearman|Manhattan Pearson|Manhattan Spearman|Dot Pearson|Dot Spearman|
|:------------------------:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|KoSBERT_SKT*|78.81|78.47|77.68|77.78|77.71|77.83|75.75|75.22|
|KoSimCSE_SKT|0|0|0|0|0|0|0|0|
 - \*: [KoSBERT_SKT](https://github.com/BM-K/KoSentenceBERT_SKT)
## Example Downstream Task
### Semantic Search
```python
```
<br> Result :
```
```

## Citing
### KorNLU Datasets
```bibtex
@article{ham2020kornli,
  title={KorNLI and KorSTS: New Benchmark Datasets for Korean Natural Language Understanding},
  author={Ham, Jiyeon and Choe, Yo Joong and Park, Kyubyong and Choi, Ilji and Soh, Hyungjoon},
  journal={arXiv preprint arXiv:2004.03289},
  year={2020}
}
```
### SimCSE
```bibtex
@article{gao2021simcse,
   title={{SimCSE}: Simple Contrastive Learning of Sentence Embeddings},
   author={Gao, Tianyu and Yao, Xingcheng and Chen, Danqi},
   journal={arXiv preprint arXiv:2104.08821},
   year={2021}
}
```
