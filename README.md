# En français si vous plait? 🇨🇦

**This is a machine learning natural language processing (NLP) project submission for the [Global PyTorch Summer Hackathon! #PTSH19](https://pytorch.devpost.com/)**. Pour la documentation en français, [cliquez ici!](https://github.com/lucylow/en_francais_si_vous_plait-/blob/master/README-fr.md)

<div>
  
  [![Status](https://img.shields.io/badge/status-active-success.svg)]()
  [![GitHub Issues](https://img.shields.io/github/issues/lucylow/en_francais_si_vous_plait-.svg)](https://github.com/lucylow/en_francais_si_vous_plait-/issues)
  [![GitHub Pull Requests](https://img.shields.io/github/issues-pr/lucylow/en_francais_si_vous_plait-.svg)](https://github.com/lucylow/en_francais_si_vous_plait-/pulls)
  [![License](https://img.shields.io/bower/l/bootstrap)]()

</div>


---

## Motivation

* Implementation of **Fairseq's Machine Learning Sequence Modeling toolkit in PyTorch**
* Machine language translation transformer model from [***Attention Is All You Need***](https://arxiv.org/abs/1706.03762) using **encoder-decoder attention mechanisms in a sequence-to-sequence model** that features stacked self attention layers
* Business applications to learn the tone of customer's communications and responds with an appropriate tone


---  

## Technical Tools

* [**Pytorch**](https://pytorch.org) 
  * Open source deep learning research platform that provides maximum flexibility and speed and provides tensors that live on the GPU accelerating the computation
  
* [**Facebook Research's Fairseq**](https://ai.facebook.com/tools/fairseq/) 
  * Sequence modeling toolkit written in PyTorch
  * Train custom models for **Neural Machine Translation (NMT)** - translation, summarization, language modeling, and other text generation tasks

* [**Transformer Machine Learning Model**](https://arxiv.org/pdf/1706.03762.pdf)
  * Transduction model relying on **self-attention layers to compute input and output represenations** where the attention functions maps [query, key-value pairs] to vector-outputs of [query, key-value pairs] - requires sequence-aligned RNNs or CNNs
  
      ![transformer model](https://github.com/lucylow/En_francais_si_vous_plait-/blob/master/screenshots/Transformer-smaller-pic.png)

    *Image of Transformer model. The encoder maps sequence X_n (x_1, x_2 ... x_n) --> sequence Z_n (z_1, z_2 ... z_n). From Z_n, the decoder generates sequence Y_n (y_1, y_2 ... y_n) element by element. [Image Source](https://arxiv.org/pdf/1706.03762.pdf)*

 
--- 

## Convolutional Self-Attention Transformer Modelling 
  
* Measure speed translations
  * Record the translation time once machine learning system is shown a sentence to quantify results
  * "On the **WMT 2014 English-to-French translation task** (a widely used metric for judging the accuracy of machine translation), attention model establishes a BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature"
  
     ![alt text](https://github.com/lucylow/En_francais_si_vous_plait-/blob/master/screenshots/Transformer%20BLEU%20scores%20Training%20Cost.png)
     
     *Image. Transformer model high in BLEU scale and low on training costs [Image Source](https://arxiv.org/pdf/1706.03762.pdf)*
  
* Gating to control flow of hidden-units
  
* **Multi-Hop Attention Functionality** 
  * Self attention layers - where all the keys, values, and queries come from the same input
  * CNN encoder creates a vector for each word to be translated, and CNN decoder translates words while PyTorch computations are being simultaneously made
  * **Network has two decoder layers and attention is paid to each layer.** Refer to image below.

      ![alt text bonjour](https://github.com/lucylow/En_francais_si_vous_plait-/blob/master/screenshots/translation_illustration.gif)

       *Image of **Multi-hop Attention** tensor computations where green lines represent attention paid to each French word. [Image Source](https://engineering.fb.com/ml-applications/a-novel-approach-to-neural-machine-translation)*


--- 
 
## French-English Translation Dataset

* Statistical machine translation [WMT 2014 French-English](http://statmt.org/wmt14/translation-task.html#Download) with **corpus size 2.3GB and 36 million sentence pairs**
* Dataset too big to include in repo - **download and extract to /data/iwslt14/** to replace iwslt14.en.txt and iwslt14.fr.txt
* For French-English translations, order of words matter and and the number of words can be added during the translation.
  * Ex1 "chat noir" vs "black cat" 
  * Ex2 "not" vs "ne ___ pas"

    ![sentences](https://github.com/lucylow/En_francais_si_vous_plait-/blob/master/screenshots/sequence2equence_%20encoderdecoder.png)
    
    *Image of **sentence sequence prediction**.[Image Source](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#sphx-glr-intermediate-seq2seq-translation-tutorial-py)*
    
* Dataset includes:
  * Commoncrawl
  * Europarl-v7
  * Giga
  * News-commentary
  * Undoc
* Pre-process WMT2014 text corpus

```terminal
cd data/
bash prepare-iwslt14.sh

TEXT=data/iwslt14.tokenized.fr-en

# Binarize data
$ fairseq preprocess -sourcelang fr -targetlang en \
    -trainpref $TEXT/train -validpref $TEXT/valid -testpref $TEXT/test \
    -thresholdsrc 3 -thresholdtgt 3 -destdir data-bin/iwslt14.tokenized.fr-en
    -workers 60
```


---

## Technical Train the French-English Model

**Train new CNN model (dropout rate of 0.2) with *-fairseq train***

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

> Je ne crains pas de mourir.

Source: Je ne crains pas de mourir.
Original_Sentence: Je ne crains pas de mourir.
Hypothesis: -0.23804219067097 I am not scared of dying.
Attention_Maxima: 2 2 3 4 5 6 7 8 9
Hypothesis: -0.23861141502857 I am not scared of dying.
Attention_Maxima: 2 2 3 4 5 7 6 7 9 9
```

---

## Visualizing Attention 

* **Step by step visualization of the encoder-decoder network attention matrix** as it goes through a sentance translation 
* Use matplotlib library to display matrix via *plt.matshow(attention)* :

  ![attention matrix](https://github.com/lucylow/En_francais_si_vous_plait-/blob/master/screenshots/attention_matrix.png)
  
  *Image of attention matrix. Input steps vs output steps with the sample sentece "Je ne crains pas de mourir."*


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
* Fairseq Technical Documentation: https://fairseq.readthedocs.io/en/latest/models.html#module-fairseq.models.transformer
* "FAIRSEQ: A Fast, Extensible Toolkit for Sequence Modeling" https://arxiv.org/pdf/1904.01038.pdf
* Neural Machine Translation by Jointly Learning to Align and Translate https://arxiv.org/abs/1409.0473
* Translation with Sequence to Sequence Network and Attention https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#sphx-glr-intermediate-seq2seq-translation-tutorial-py
* "Convolutional Sequence to Sequence Learning" https://arxiv.org/abs/1705.03122
* "Attention Is All You Need" https://arxiv.org/abs/1706.03762
* Data processing scripts: https://www.dagshub.com/Guy/fairseq/src/67af40c9cca0241d797be13ae557d59c3732b409/data
* Beyond "How May I Help You?": https://medium.com/airbnb-engineering/beyond-how-may-i-help-you-fd6a0d385d02

