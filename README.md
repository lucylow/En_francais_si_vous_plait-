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

*Image of **Multi-hop Attention** computations from [here](https://engineering.fb.com/ml-applications/a-novel-approach-to-neural-machine-translation) where green lines represent attention paid to each French word.*

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

