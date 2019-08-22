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

* **Linguistic analysis** to detect language tones from written text
* Implementation of **Fairseq's Machine Learning Sequence Modeling toolkit in PyTorch**
* Business applications to learn the tone of customer's communications and responds with an appropriate tone


---

## French-English Translations
* Masculine “the” (le) vs Feminine “the” (la)
* Masculine “a” (un) vs Feminine “a” (une)
* Passé Composé Tense: 
  * Je suis allé(e) 
  * Tu es allé(e) 
  * Il/elle est allé(e) 
  * Nous sommes allé(e)s 
  * Vous êtes allé(e)(s) 
  * Ils/elles sont allé(e)s
    
---  

## Technical Tools
* [**Pytorch**](https://pytorch.org) 
  * Deep learning research platform that provides maximum flexibility and speed
  * Provides Tensors that can live either on the CPU or the GPU, and accelerates the computation by a huge amount
  
* [**Facebook AI Research's Fairseq**](https://ai.facebook.com/tools/fairseq/) 
  * Sequence modeling toolkit written in PyTorch
  * Train custom models for translation, summarization, language modeling, and other text generation tasks
 
--- 
 

## Technical Convolutional Neural Networks (CNN)

* Record speed translations
  * Measure translation time once machine learning system is shown a sentence
  * "**The CNN outperforms it by 1.5 BLEU on the WMT 2014 English-French task**, a widely used metric for judging the accuracy of machine translation."
  
* Gating to control flow of hidden-units

* **Multi-Hop Attention** 
  * CNN encoder creates a vector for each word to be translated
  * CNN decoder translates the English words while computations are being simultaneously made
  * Network has two decoder layers and attention is paid to each layer. Refer to image below.

      ![alt text bonjour](https://github.com/lucylow/En_francais_si_vous_plait-/blob/master/screenshots/translation_illustration.gif)

       *Image of **Multi-hop Attention** tensor computations from [here](https://engineering.fb.com/ml-applications/a-novel-approach-to-neural-machine-translation) where green lines represent attention paid to each French word.*

---

# Technical Translation Model and Test Sets

* Fully pretrained model
  * wmt14.en-fr.fconv-cuda.tar.bz2: Pre-trained model for WMT14 English-French including vocabularies
  
* Test sets for model
  * wmt14.en-fr.newstest2014.tar.bz2: newstest2014 test set for WMT14 English-French
  * wmt14.en-fr.ntst1213.tar.bz2: newstest2012 and newstest2013 test sets for WMT14 English-French

---

# Pre-trained Models

1) Download French-English model wmt14.en-fr.fconv-cuda/

  > $ curl https://s3.amazonaws.com/fairseq/models/wmt14.en-fr.fconv-cuda.tar.bz2 | tar xvjf -

2) Translate text with *fairseq generate-lines*

'''python

> Why is it rare to discover new marine mam@@ mal species ?
Source:	Why is it rare to discover new marine mam@@ mal species ?
Original_Sentence:	Why is it rare to discover new marine mam@@ mal species ?
Hypothesis:	-0.068684287369251	Pourquoi est-il rare de découvrir de nouvelles espèces de mammifères marins ?
Attention_Maxima:	1 1 4 4 6 6 7 11 9 9 9 12 13

'''


---

# Train New Language Model

1) Pre-process the French-English corpus on terminal

> cd data/

> bash prepare-iwslt14.sh

> cd ..

> TEXT=data/iwslt14.tokenized.fe-en

> $ fairseq preprocess -sourcelang fr -targetlang en \
    -trainpref $TEXT/train -validpref $TEXT/valid -testpref $TEXT/test \
    -thresholdsrc 3 -thresholdtgt 3 -destdir data-bin/iwslt14.tokenized.fr-en


2) Train new CNN model with *fairseq train* (uses all GPUs on machine)

'''
$ mkdir -p trainings/fconv
$ fairseq train -sourcelang de -targetlang en -datadir data-bin/iwslt14.tokenized.de-en \
  -model fconv -nenclayer 4 -nlayer 3 -dropout 0.2 -optim nag -lr 0.25 -clip 0.1 \
  -momentum 0.99 -timeavg -bptt 0 -savedir trainings/fconv
'''

3) Model Generation with  *fairseq generate*

'''python
$ DATA=data-bin/iwslt14.tokenized.fe-en
$ fairseq generate-lines -sourcedict $DATA/dict.fe.th7 -targetdict $DATA/dict.en.th7 \
  -path trainings/fconv/model_best_opt.th7 -beam 10 -nbest 2
| [target] Dictionary: 24738 types
| [source] Dictionary: 35474 types
> Pourquoi est-il rare de découvrir de nouvelles espèces de mammifères marins ?
Source:	Pourquoi est-il rare de découvrir de nouvelles espèces de mammifères marins ?
Original_Sentence:	Pourquoi est-il rare de découvrir de nouvelles espèces de mammifères marins ?
Hypothesis:	-0.23804219067097	Why is it rare to discover new marine mam@@ mal species ?
A	2 2 3 4 5 6 7 8 9
Hypothesis:	-0.23861141502857	Why is it rare to discover new marine mam@@ mal species ?
Attention_Maxima:	2 2 3 4 5 7 6 7 9 9
'''



---

## References
* https://ai.facebook.com/tools/fairseq/
* "FAIRSEQ: A Fast, Extensible Toolkit for Sequence Modeling" https://arxiv.org/pdf/1904.01038.pdf
* "Convolutional Sequence to Sequence Learning" https://arxiv.org/abs/1705.03122
* "Attention Is All You Need" https://arxiv.org/abs/1706.03762

