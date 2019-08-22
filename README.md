# En français si vous plait? &#x1F536;

Pour la documentation en français, [cliquez ici 🇨🇦](https://github.com/lucylow/en_francais_si_vous_plait/blob/master/README-fr.md)

<div>
  
  [![Status](https://img.shields.io/badge/status-work--in--progress-success.svg)]()
  [![GitHub Issues](https://img.shields.io/github/issues/lucylow/en_francais_si_vous_plait.svg)](https://github.com/lucylow/en_francais_si_vous_plait/issues)
  [![GitHub Pull Requests](https://img.shields.io/github/issues-pr/lucylow/en_francais_si_vous_plait.svg)](https://github.com/lucylow/en_francais_si_vous_plait/pulls)
  [![License](https://img.shields.io/bower/l/bootstrap)]()

</div>

---

## Intro &#x1F536;

* **Linguistic analysis to detect language tones from written text**
* Analyze tone on document and sentence levels for **French-English translations**
* Learn the tone of customer's communications and responds with an appropriate tone
  

## Problem / Motivation &#x1F536;
“Mais non, le masculin l’emporte sur le féminin!” == **“But no, the masculine takes precedence over the feminine.**”

  ![Famille](https://github.com/lucylow/en_francais_si_vous_plait/blob/master/famille.png)

*Image Reference: Grade 4 Lucy Low 🇨🇦*

## French-English Examples &#x1F536;
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


## IBM Watson's Tone Analyzer
* **Linguistic analysis** to detect language tones from written text
* Requires API authentication via token-based identity
* Input content **128KB of data** or **1000 indivdual sentences** in plain text (the ASCII character set) 
* For more info see here: https://cloud.ibm.com/apidocs/tone-analyzer

  ![IBM sentence tone data structure](https://github.com/lucylow/en_francais_si_vous_plait/blob/master/sentence%20tones.png)

*Image of data structure for sentence tones*

## Masculine Endings &#x1F538;

-an, -and, -ant, -ent, -in, -int, -om, -ond, -ont, -on, -eau, -au, -aud, -aut, -o, -os, -ot -ai, -ais, -ait, -es, -et -ou, -out, -out, -oux, -i, -il, -it, -is, -y -at, -as, -ois, -oit, -u, -us, -ut, -eu, -er, -é, -age, -ege, – ème, -ome, -aume, -isme -as, -is, -os, -us, -ex, -it, -est, -al, -el, -il, -ol, -eul, -all, -if, -ef, -ac, -ic, -oc, -uc -am, -um, -en, -air, -er, -erf, -ert, -ar, -arc, -ars, -art, -our, -ours, -or, -ord, -ors, -ort, -ir, -oir, -eur, -ail, -eil, -euil, -ueil, -ing


## Feminine Endings &#x1F538;

-aie, -oue, -eue, -ion, -te, – ée, -ie, -ue, -asse, -ace, -esse, -ece, -aisse, -isse, -ice, -ousse, -ance, -anse, -ence, -once -enne, -onne, -une, -ine, -aine, -eine, -erne, -ande, -ende, -onde, -ade, -ude, -arde, -orde, -euse, -ouse, -ase, -aise, -ese, -oise, -ise, -yse, -ose, -use, -ache, -iche, -eche, -oche, -uche, -ouche, -anche, -ave, -eve, -ive, -iere, -ure, -eure, -ette, -ete, – ête, -atte, -otte, -oute, -orte, -ante, -ente, -inte, -onte, -alle, -elle, -ille, -olle, -aille, -eille, -ouille, -appe, -ampe, -ombe, -igue


## Tools &#x1F536;

* Django Python Framework
* IBM Watson Tone Analyzer V3

## References &#x1F536;

* Caroline Criado-Pérez. "Invisble Women: Data Bias in a World Designed for Men."
