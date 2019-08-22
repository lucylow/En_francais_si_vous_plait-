# En français si vous plait? 🇨🇦

Pour la documentation en français, [cliquez ici!](https://github.com/lucylow/en_francais_si_vous_plait-/blob/master/README-fr.md)

<div>
  
  [![Status](https://img.shields.io/badge/status-active-success.svg)]()
  [![GitHub Issues](https://img.shields.io/github/issues/lucylow/en_francais_si_vous_plait-.svg)](https://github.com/lucylow/en_francais_si_vous_plait-/issues)
  [![GitHub Pull Requests](https://img.shields.io/github/issues-pr/lucylow/en_francais_si_vous_plait-.svg)](https://github.com/lucylow/en_francais_si_vous_plait-/pulls)
  [![License](https://img.shields.io/bower/l/bootstrap)]()

</div>

**This is a machine learning project submission for the [Global PyTorch Summer Hackathon! #PTSH19](https://pytorch.devpost.com/)**. French-English linguistic analysis to detect language tones from written text on document and sentence levels.

---

## Motivation/Introduction

* **Linguistic analysis to detect language tones from written text**. Learn the tone of customer's communications and responds with an appropriate tone  
* Fairseq Machine Learning Sequence Modeling toolkit in PyTorch
* Record speed translations (how long it takes to get a translation once we show the system a sentence) : "The **CNN outperforms it by 1.5 BLEU on the WMT 2014 English-French task**, a widely used metric for judging the accuracy of machine translation."
* **Multi-Hop Attention** for faster language encoding and decoding 
* Gating to control flow of hidden-units in neuural network

![alt text bonjour](https://github.com/lucylow/En_francais_si_vous_plait-/blob/master/screenshots/translation_illustration.gif)
*Image of **Multi-hop Attention computations** from https://engineering.fb.com/ml-applications/a-novel-approach-to-neural-machine-translation/. CNN encoder creates a vector for each word to be translated. CNN decoder trnslates the English words while computations are being simultaneously made. Network has two layers in decoder and attention is paid to each layer. Greenline == attention fo each French word. *



*
*


---

## French-English Examples
1. masculine “the” (le) vs feminine “the” (la)
2. masculine “a” (un) vs feminine “a” (une)
3. masculine "du" vs feminine “de la”
4. masculine "knife" vs feminine "fork"
5. masculine "madame le ministre" vs feminine "madame la ministre"
6. masculine "directeur" vs feminine "directrice" **but mixed genders is "directeurs"**
7. passé composé tense: 
    * je suis allé(e) 
    * tu es allé(e) 
    * il/elle est allé(e) 
    * nous sommes allé(e)s 
    * vous êtes allé(e)(s) 
    * ils/elles sont allé(e)s

## Masculine Endings

-an, -and, -ant, -ent, -in, -int, -om, -ond, -ont, -on, -eau, -au, -aud, -aut, -o, -os, -ot -ai, -ais, -ait, -es, -et -ou, -out, -out, -oux, -i, -il, -it, -is, -y -at, -as, -ois, -oit, -u, -us, -ut, -eu, -er, -é, -age, -ege, – ème, -ome, -aume, -isme -as, -is, -os, -us, -ex, -it, -est, -al, -el, -il, -ol, -eul, -all, -if, -ef, -ac, -ic, -oc, -uc -am, -um, -en, -air, -er, -erf, -ert, -ar, -arc, -ars, -art, -our, -ours, -or, -ord, -ors, -ort, -ir, -oir, -eur, -ail, -eil, -euil, -ueil, -ing


## Feminine Endings

-aie, -oue, -eue, -ion, -te, – ée, -ie, -ue, -asse, -ace, -esse, -ece, -aisse, -isse, -ice, -ousse, -ance, -anse, -ence, -once -enne, -onne, -une, -ine, -aine, -eine, -erne, -ande, -ende, -onde, -ade, -ude, -arde, -orde, -euse, -ouse, -ase, -aise, -ese, -oise, -ise, -yse, -ose, -use, -ache, -iche, -eche, -oche, -uche, -ouche, -anche, -ave, -eve, -ive, -iere, -ure, -eure, -ette, -ete, – ête, -atte, -otte, -oute, -orte, -ante, -ente, -inte, -onte, -alle, -elle, -ille, -olle, -aille, -eille, -ouille, -appe, -ampe, -ombe, -igue

---

## Technical Tools

* [Facebook AI Research's Fairseq](https://ai.facebook.com/tools/fairseq/) 
  * Sequence modeling toolkit written in PyTorch
  * Train custom models for translation, summarization, language modeling and other text generation tasks.


# Steps

1) Install PyTorch
2) Install fairseq-py

  > git clone https://github.com/pytorch/fairseq.git
  > cd fairseq
  > pip install -r requirements.txt
  > python setup.py build develop

3) Download and train ML model


## References
- https://ai.facebook.com/tools/fairseq/
- "FAIRSEQ: A Fast, Extensible Toolkit for Sequence Modeling" https://arxiv.org/pdf/1904.01038.pdf
- "Convolutional Sequence to Sequence Learning" https://arxiv.org/abs/1705.03122

