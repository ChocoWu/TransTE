# Transition-based end-to-end Triplet Extraction
Source code for Neurocomputing paper:[Neural Transition Model for Aspect-based Sentiment Triplet Extraction with Triplet Memory](https://www.sciencedirect.com/science/article/abs/pii/S0925231221011887)
## Dependencies:
+ DyNET 2.1
+ Pyyaml 5.1
+ gensim 3.7.3
+ Pytorch 1.1.0
+ pytorch-transformer 1.2.0
+ flair 0.4.3
+ AllenNLP

## Dataset:
See example data `data/triplet/14lap/train.json`.

```angular2
Drivers updated ok but the BIOS update froze the system up and the computer shut down .####Drivers=T-POS updated=O ok=O but=O the=O BIOS=TT-NEG update=TT-NEG froze=O the=O system=TTT-NEG up=O and=O the=O computer=O shut=O down=O .=O####Drivers=O updated=O ok=S but=O the=O BIOS=O update=O froze=SS the=O system=O up=O and=O the=O computer=O shut=O down=O .=O####[([0], [2], 'POS'), ([5, 6], [7], 'NEG'), ([9], [7], 'NEG')]
```
Each line includes four parts which seperate by '####':
- raw input sentence.
- Aspect terms with sentiment polarity which are labeled with 'O/T/TT/TTT'.
- Opinion term which are labeled with 'O/S/SS/SSS'.
- Triplets which is formed as '[(aspect term, opinion term, polarity)]' 


## Configurations 
* `data_config.yaml` (for locating file paths)  
* `joint_config.yaml` (for parameters tuning)

## Preprocess for preparing ready-to-go data:
+ Put glove.6B.100d.txt in `embedding_dir` which is set in the `data_config.yaml`

+ Then make vocabulary and pickle instances by (Note: we employ *[Corenlp](https://stanfordnlp.github.io/CoreNLP/)* to obtain the dependency tree and POS tags): 
```
python preprocess.py
```

+ (Optional) Generate BERT Embeddings (bert-base-uncased): 
```
python gen_bert_emb.py
```
Note that if you don\`t use BERT, set `use_sentence_vec` to false in `joint_config.yaml`.


## Train & Evaluate:
```
python train.py
```
