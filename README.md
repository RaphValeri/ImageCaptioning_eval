Image Captioning Evaluation
==========================

This repository has been forked from the [pycocoevalcap](https://github.com/salaniz/pycocoevalcap) repository. The developed scripts added to this initial repository are `metrics_compute.py` and `visualize_captions.py`. To use these, you need to add a 'res_file' folder with all the JSON results files inside organised by foler. 
For instance if you used a captioning model with 1 Cross-Attention layer trained on 3 epochs, you need to add a 'res_files/1ca_ep3' folder and add the JSON file inside.


Evaluation codes for MS COCO caption generation.

## Description ##
This repository provides Python 3 support for the caption evaluation metrics used for the MS COCO dataset.

The code is derived from the original repository that supports Python 2.7: https://github.com/tylin/coco-caption.  
Caption evaluation depends on the COCO API that natively supports Python 3.

## Requirements ##
- Java 1.8.0
- Python 3


## Usage ##
Run the following script: [metrics_compute.py](./metrics_compute.py)

## Added files ##
./
- metrics_compute.py : script generating the metrics computations based on the JSON files in the added folders as explained above
- visualize_captions.ipynb : a jupyter-notebook files aiming to display some example captions from a specific JSON file 


## References ##

- [Microsoft COCO Captions: Data Collection and Evaluation Server](http://arxiv.org/abs/1504.00325)
- PTBTokenizer: We use the [Stanford Tokenizer](http://nlp.stanford.edu/software/tokenizer.shtml) which is included in [Stanford CoreNLP 3.4.1](http://nlp.stanford.edu/software/corenlp.shtml).
- BLEU: [BLEU: a Method for Automatic Evaluation of Machine Translation](http://www.aclweb.org/anthology/P02-1040.pdf)
- Meteor: [Project page](http://www.cs.cmu.edu/~alavie/METEOR/) with related publications. We use the latest version (1.5) of the [Code](https://github.com/mjdenkowski/meteor). Changes have been made to the source code to properly aggreate the statistics for the entire corpus.
- Rouge-L: [ROUGE: A Package for Automatic Evaluation of Summaries](http://anthology.aclweb.org/W/W04/W04-1013.pdf)
- CIDEr: [CIDEr: Consensus-based Image Description Evaluation](http://arxiv.org/pdf/1411.5726.pdf)
- SPICE: [SPICE: Semantic Propositional Image Caption Evaluation](https://arxiv.org/abs/1607.08822)

## Developers ##
- Xinlei Chen (CMU)
- Hao Fang (University of Washington)
- Tsung-Yi Lin (Cornell)
- Ramakrishna Vedantam (Virgina Tech)

## Acknowledgement ##
- David Chiang (University of Norte Dame)
- Michael Denkowski (CMU)
- Alexander Rush (Harvard University)
