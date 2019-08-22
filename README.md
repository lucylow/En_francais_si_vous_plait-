# En français si vous plait? 🇨🇦

**This is a machine learning project submission for the [Global PyTorch Summer Hackathon! #PTSH19](https://pytorch.devpost.com/)**. Pour la documentation en français, [cliquez ici!](https://github.com/lucylow/en_francais_si_vous_plait-/blob/master/README-fr.md)

<div>
  
  [![Status](https://img.shields.io/badge/status-active-success.svg)]()
  [![GitHub Issues](https://img.shields.io/github/issues/lucylow/en_francais_si_vous_plait-.svg)](https://github.com/lucylow/en_francais_si_vous_plait-/issues)
  [![GitHub Pull Requests](https://img.shields.io/github/issues-pr/lucylow/en_francais_si_vous_plait-.svg)](https://github.com/lucylow/en_francais_si_vous_plait-/pulls)
  [![License](https://img.shields.io/bower/l/bootstrap)]()

</div>


---

## Motivation

* Implementation of **Fairseq's Machine Learning Sequence Modeling toolkit in PyTorch**
* Machine language translation transformer model from [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762)
* Business applications to learn the tone of customer's communications and responds with an appropriate tone

    
---  

## Technical Tools

* [**Pytorch**](https://pytorch.org) 
  * Deep learning research platform that provides maximum flexibility and speed and provides tensors that live on the GPU accelerating the computation
  
* [**Facebook Research's Fairseq**](https://ai.facebook.com/tools/fairseq/) 
  * Sequence modeling toolkit written in PyTorch
  * Train custom models for **Neural Machine Translation (NMT)** - translation, summarization, language modeling, and other text generation tasks
 

--- 

## Convolutional Transformer Sequence-to-Sequence Modeling 

* Measure speed translations
  * Record the translation time once machine learning system is shown a sentence to quantify results
  * "**The CNN outperforms it by 1.5 BLEU on the WMT 2014 French-English task**, a widely used metric for judging the accuracy of machine translation."
  
* Gating to control flow of hidden-units

* **Multi-Hop Attention** 
  * CNN encoder creates a vector for each word to be translated, and CNN decoder translates words while PyTorch computations are being simultaneously made
  * **Network has two decoder layers and attention is paid to each layer.** Refer to image below.

      ![alt text bonjour](https://github.com/lucylow/En_francais_si_vous_plait-/blob/master/screenshots/translation_illustration.gif)

       *Image of **Multi-hop Attention** tensor computations where green lines represent attention paid to each French word. [Image Source](https://engineering.fb.com/ml-applications/a-novel-approach-to-neural-machine-translation)*


--- 
 
## French-English Translation Dataset

* Statistical machine translation [WMT 2014 French-English](http://statmt.org/wmt14/translation-task.html#Download) with **corpus size 2.3GB and 40.8M sentences**
* Dataset includes:
  * Commoncrawl
  * Europarl-v7
  * Giga
  * News-commentary
  * Undoc
* Pre-process WMT2014 text corpus

```terminal
bash prepare-iwslt14.sh

TEXT=iwslt14.tokenized.fr-en

# Binarize data
$ fairseq preprocess -sourcelang fr -targetlang en \
    -trainpref $TEXT/train -validpref $TEXT/valid -testpref $TEXT/test \
    -thresholdsrc 3 -thresholdtgt 3 -destdir data-bin/iwslt14.tokenized.fr-en
    -workers 60
```

---

## Technical Train the French-English Model

**Train new CNN model with *-fairseq train***

```python
$ mkdir -p trainings/fconv

$ fairseq train -sourcelang fr -targetlang en -datadir data-bin/iwslt14.tokenized.fr-en \
  -model fconv -nenclayer 4 -nlayer 3 -dropout 0.2 -optim nag -lr 0.25 -clip 0.1 \
  -momentum 0.99 -timeavg -bptt 0 
  -savedir trainings/fconv
```

**Model Generation with *-fairseq generate***

```python
$ DATA=data-bin/iwslt14.tokenized.fr-en

$ fairseq generate-lines -sourcedict $DATA/dict.fr.th7 -targetdict $DATA/dict.en.th7 \
  -path trainings/fconv/model_best_opt.th7 -beam 10 -nbest 
| [target] Dictionary: 24738 types
| [source] Dictionary: 35474 types

> Pourquoi est-il rare de découvrir de nouvelles espèces de mammifères marins ?

Source: Pourquoi est-il rare de découvrir de nouvelles espèces de mammifères marins ?
Original_Sentence: Pourquoi est-il rare de découvrir de nouvelles espèces de mammifères marins ?
Hypothesis:-0.23804219067097 Why is it rare to discover new marine mam@@ mal species ?
Attention_Maxima: 2 2 3 4 5 6 7 8 9
Hypothesis:-0.23861141502857 Why is it rare to discover new marine mam@@ mal species ?
Attention_Maxima: 2 2 3 4 5 7 6 7 9 9
```
---

##  Technical Models and Test Sets

* Fully pretrained model
  * **wmt14.en-fr.fconv-cuda.tar.bz2:** Pre-trained model for WMT14 English-French including vocabularies
  
* Test sets for model
  * **wmt14.en-fr.newstest2014.tar.bz2:** newstest2014 test set for WMT14 English-French
  * **wmt14.en-fr.ntst1213.tar.bz2:** newstest2012 and newstest2013 test sets for WMT14 English-French


---

## References

* https://ai.facebook.com/tools/fairseq/
* "FAIRSEQ: A Fast, Extensible Toolkit for Sequence Modeling" https://arxiv.org/pdf/1904.01038.pdf
* "Convolutional Sequence to Sequence Learning" https://arxiv.org/abs/1705.03122
* "Attention Is All You Need" https://arxiv.org/abs/1706.03762
* Data processing scripts: https://www.dagshub.com/Guy/fairseq/src/67af40c9cca0241d797be13ae557d59c3732b409/data

