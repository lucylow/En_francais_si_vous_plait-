# En français si vous plait? 🇨🇦

**Ceci est une soumission de projet d'apprentissage automatique pour le [Global PyTorch Summer Hackathon! # PTSH19](https://pytorch.devpost.com/)**. 
For the English readme documentation, [click here!](https://github.com/lucylow/en_francais_si_vous_plait-/blob/master/README.md)

<div>
  
  [![Status](https://img.shields.io/badge/status-active-success.svg)]()
  [![GitHub Issues](https://img.shields.io/github/issues/lucylow/en_francais_si_vous_plait.svg)](https://github.com/lucylow/en_francais_si_vous_plait/issues)
  [![GitHub Pull Requests](https://img.shields.io/github/issues-pr/lucylow/en_francais_si_vous_plait.svg)](https://github.com/lucylow/en_francais_si_vous_plait/pulls)
  [![License](https://img.shields.io/bower/l/bootstrap)]()

---

## Motivational

* Mise en œuvre de la boîte à outils **de Fairseq, Machine Learning Sequence Modeling, dans PyTorch**
* Modèle de transformateur de traduction en langage machine de [*L'attention est tout ce dont vous avez besoin*](https://arxiv.org/abs/1706.03762)
* Applications métier pour connaître le ton des communications du client et répondre avec un ton approprié

---

## Outils Techniques

* [**Pytorch**](https://pytorch.org)
  * Plateforme de recherche en apprentissage en profondeur offrant une flexibilité et une vitesse maximales ainsi que des tenseurs basés sur le processeur graphique accélérant le calcul
  
* [**Fairseq de Facebook Research**](https://ai.facebook.com/tools/fairseq/)
  * Boîte à outils de modélisation de séquence écrite en PyTorch
   * Former des modèles personnalisés pour **traduction neuronale** - traduction, synthèse, modélisation de langage et autres tâches de génération de texte
 

---

## Transformateur Convolutionnel

* Mesurer les traductions de vitesse
  * Enregistrez le temps de traduction une fois que le système d'apprentissage machine affiche une phrase pour quantifier les résultats
  * "**CNN le surpasse de 1,5 BLEU pour la tâche français-anglais du WMT 2014**, une métrique largement utilisée pour juger de l'exactitude de la traduction automatique."
  
* Gating pour contrôler le flux d'unités cachées

* **Attention multi-hop**
  * Le codeur CNN crée un vecteur pour chaque mot à traduire et le décodeur CNN traduit les mots pendant que les calculs PyTorch sont effectués simultanément
   * **Le réseau a deux couches de décodeur et une attention particulière est accordée à chaque couche.** Voir l'image ci-dessous:

    ![alt text bonjour](https://github.com/lucylow/En_francais_si_vous_plait-/blob/master/screenshots/translation_illustration.gif)

    *Image de **Calculs tensoriels** à sauts multiples où les lignes vertes représentent l'attention portée à chaque mot français. [Source d'image](https://engineering.fb.com/ml-applications/a-novel-approach-to-neural-machine-translation)*

---
 
## Traduction Français-Anglais

* Traduction automatique statistique [WMT 2014 Français-Anglais](http://statmt.org/wmt14/translation-task.html#Download) avec **corpus de 2.3GB et 40.8 millions phrases**

* Le jeu de données comprend:
  * Commoncrawl
  * Europarl-v7
  * Giga
  * Nouvelles-commentaires
  * Undoc

* Pré-traitement du corpus de texte WMT2014

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

## Formation Technique sur le Modèle Français-Anglais

**Former le nouveau modèle CNN avec le train *-fairseq***

```python
$ mkdir -p trainings/fconv

$ fairseq train -sourcelang fr -targetlang en -datadir data-bin/iwslt14.tokenized.fr-en \
  -model fconv -nenclayer 4 -nlayer 3 -dropout 0.2 -optim nag -lr 0.25 -clip 0.1 \
  -momentum 0.99 -timeavg -bptt 0 
  -savedir trainings/fconv
```

**La Génération de Modèle avec *-fairseq génère***

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

## Modèles Techniques et Ensembles de Test

* Modèle entièrement pré-entrainé
  * **wmt14.en-fr.fconv-cuda.tar.bz2:** Modèle pré-formé pour le WMT14 anglais-français, y compris les vocabulaires
  
* Ensembles de test pour le modèle
  * **wmt14.en-fr.newstest2014.tar.bz2:** Ensemble de test newstest2014 pour WMT14 anglais-français
  * **wmt14.en-fr.ntst1213.tar.bz2:** ensembles de test newstest2012 et newstest2013 pour WMT14 anglais-français


---

## Références
* https://ai.facebook.com/tools/fairseq/
* Fairseq Technical Documentation: https://fairseq.readthedocs.io/en/latest/models.html#module-fairseq.models.transformer
* "FAIRSEQ: A Fast, Extensible Toolkit for Sequence Modeling" https://arxiv.org/pdf/1904.01038.pdf
* "Convolutional Sequence to Sequence Learning" https://arxiv.org/abs/1705.03122
* "Attention Is All You Need" https://arxiv.org/abs/1706.03762
* Data processing scripts: https://www.dagshub.com/Guy/fairseq/src/67af40c9cca0241d797be13ae557d59c3732b409/data

