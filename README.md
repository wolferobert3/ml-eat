# ML-EAT Repository
This is the code repository for the AIES'24 paper ***ML-EAT: A Multilevel Embedding Association Test  for Interpretable and Transparent Social Science***, available at [placeholder](placeholder).

### 1. Structure

This project includes three implementations of the ML-EAT. The first is a torch-based implementation, included in the mleat.py file and applied to the GloVe embeddings in the "empirical_analysis" notebook. The second is a numpy-based implementation, applied to the HistWords embeddings in "histwords_analysis", as described in the paper. The third is a simple object-oriented implementation that shows programmatically how Level 2 of the ML-EAT generalizes the SC-EAT, which is included in the "sc_eat_generalization_validation" notebook. Text stimuli are drawn from Caliskan et al., 2017, and image stimuli are drawn from Steed et al., 2021 (see below), who themselves draw on the prior work of implicit bias scholars (see, for example [https://www.projectimplicit.net/](https://www.projectimplicit.net/)).

### 2. Requirements

The requirements file includes the libraries needed to run the analyses. We recommend creating a unique environment for running the project (for example, with conda, `conda create -n "mleat" python=3.11`) and then installing the requirements (`pip install -r requirements.txt`). Torchtext will need to download embeddings for empirical analysis, which may take some time.

### 3. Paper & Citation

Please cite the following version of our paper, from the AIES proceedings:

> TBD

### 4. Other Resources

This work draws on prior research in embedding association tests, implicit bias, and computational social science more broadly. Below are a few essential resources for understanding the context of this research:

- [https://www.science.org/doi/10.1126/science.aal4230](**https://www.science.org/doi/10.1126/science.aal4230**): Caliskan et al.'s foundational work demonstrating the replication of tests of implicit bias in word embeddings using the WEAT.
- [https://github.com/ryansteed/ieat](**https://github.com/ryansteed/ieat**): The github repository for Steed et al.'s extension of the WEAT to unsupervised image embeddings such as those produced by iGPT. Refer to this for image stimuli used in the present work.
- [https://pytorch.org/text/stable/index.html](**https://pytorch.org/text/stable/index.html**): Documentation for the torchtext library, which is used extensively in this repository for empirical analysis of word embedding associations.
- [https://github.com/williamleif/histwords](**https://github.com/williamleif/histwords**): The github repository for Leif et al.'s HistWords embeddings, an great resource for CSS research that includes decade-by-decade diachronic word embeddings over 200 years in several languages.
- [https://nlp.stanford.edu/projects/glove/](**https://nlp.stanford.edu/projects/glove/**): An introduction to the GloVe embeddings with useful information about training a model on a custom corpus, and using it to compare word similarities.
- [https://github.com/openai/CLIP](**https://github.com/openai/CLIP**): The github repository for OpenAI's landmark CLIP model, which enabled the study of associations between text and images in an unsupervised model.
- [https://huggingface.co/learn/nlp-course/chapter1/1](**https://huggingface.co/learn/nlp-course/chapter1/1**): The HuggingFace NLP course, a great way to get quickly caught up on the state of the art in open language technologies like those used in this work.