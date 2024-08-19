<div align="center">

# Bert-Based E-mail Classifier

## Final Project for Technion's ECE Deep Learning Course (046211)

Gal Granot: [LinkedIn](https://www.linkedin.com/in/gal-granot/), [GitHub](https://github.com/GalGranot)

Nir Tevet: [LinkedIn](https://www.linkedin.com/in/nir-tevet-355b28229/), [GitHub](https://github.com/nirtevet)

</div>

### Getting Started

```bash
git clone https://github.com/GalGranot/BERT-EmailClassifier.git
pip install -r requirements.txt
```
```python
from src.utils import (
    classify_email,
    get_top_tokens_from_text,
    clean_email,
    example_usage_clean)
```

## Introduction
This repository contains for an email proccessing Transformer, based on the Bidirectional Encoder Representations from Transformers (BERT) language model. The model, based on the attention architecture, attempts to classify legitimate emails from malicious phishing attempts, and can be used as an additional layer of protection against these types of attacks.

This project was written as part of the [Technion's ECE Deep Learning course (046211)](https://taldatech.github.io/ee046211-deep-learning/).

We've used [this Kaggle database](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset) as a baseline for performance - it contains 82k emails with roughly equal amounts of legitimate and phishing/spam emails.

## Background

The goal of this project is to develop an effective deep learning model for phishing email detection. By leveraging transformer models, we aim to accurately classify emails and highlight key words that contribute most to the classification decision.

Phishing attacks continue to be a significant cybersecurity threat, targeting both individuals and organizations. Traditional rule-based systems often are not enough to defend the average email user from the evolving nature of phishing techniques. Machine learning models, particularly large language models based on transformers, offer a promising solution due to their ability to capture contextual information in text. Furthermore, enhancing model explainability and providing transparency into the decision-making process as seen by the user is crucial for gaining the latter’s trust and ensuring that the model's decisions can be validated.

We employed a pretrained BERT model for our phishing detection task. BERT is a transformer-based model designed to capture deep bidirectional context of each token by jointly conditioning on both left and right context in all layers. This enables BERT to capture rich contextual information, making it ideal for text classification tasks.

Our workflow consisted of using the pre-trained BERT model with an added classification layer and training it (while maintaining some weights as no-decay) on the email database. After conducting the hyperparameter search, we trained the optimized model on the dataset, validated it during training using the validation set, while trying not to overfit the model in order to improve generalization.

## Results

 Our model achieved an accuracy rate of 98% on the test dataset, in addition to highlighting critical words that contributed to the model’s decision-making process. We've also visualized the attention mechanism using [BertViz](https://github.com/jessevig/bertviz).

### Attention visualization:

<div align="center">
<img src="images/purple.png" alt="Attention visualization" width="550"/>

<img src="images/yellow.png" alt="Attention visualization" width="550"/>

<img src="images/red.png" alt="Attention visualization" width="550"/>


<img src="images/green.png" alt="Attention visualization" width="550"/>
</div>

### Training & Validation Loss:
<div align="center">
  
<img src="images/loss.jpeg" alt="Attention visualization" width="800"/>

</div>

## Confusion Matrices:

<div align="center">
  
<img src="images/conf1.jpeg" alt="Attention visualization" width="550"/>

<img src="images/conf2.jpeg" alt="Attention visualization" width="550"/>

</div>

Out of 1312 datapoints in the test dataset, the model successfully classified 1286 datapoints. The false-negative and false-positive rates are very comparable (11 false-negatives and 15 false-positives), which suggests strong confidence in the predictions as well as a strong generalization ability.


## How to run

```python
email = '''
Hello, welcome to the Bert-Based E-mail Classifier! This is an example 
usage for the classify_email method. We'll clean this email and classify it
'''
print(classify_email(email))
```

## File Overview

| Directory | File | Usage |
|-----------|------|---------|
| . | README.md | this file
| . | requirements.txt | pip install requirements file | 
| . | Phishing_Email.csv | Kaggle email database |
| research | project.ipynb | notebook interface for running project |
| research | project_raw.ipynb | our process of developing the project |
| src | utils.py | APIs for classifying and preprocessing emails |
| src | train.py | training model functions |
| model | . | model contents |


## Steps to Set Up the Project

### Installation Commands From Command Line

| Step                      | Command                                                |
|---------------------------|--------------------------------------------------------|
| Install `torchdata`        | `pip install torchdata`                                |
| Install `portalocker`      | `pip install portalocker`                              |
| Install `kaggle`           | `pip install kaggle`                                   |
| Set up `kaggle.json`       | `mkdir -p ~/.kaggle` <br> `cp kaggle.json ~/.kaggle/` <br> `chmod 600 ~/.kaggle/kaggle.json` |
| Download dataset           | `kaggle datasets download -d subhajournal/phishingemails` |
| Install `bertviz`          | `pip install bertviz`                                  |
| Install `jupyterlab`      | `pip install jupyterlab`                               |
| Install `ipywidgets`      | `pip install ipywidgets`                               |
| Unzip dataset              | `unzip /content/phishingemails.zip`                    |


## Libraries and Versions

| Library                                    | Version   |
|--------------------------------------------|-----------|
| `numpy`                                    | `1.25.0`  |
| `pandas`                                   | `2.1.0`   |
| `matplotlib`                               | `3.7.2`   |
| `torch`                                    | `2.0.1`   |
| `scikit-learn`                             | `1.2.2`   |
| `transformers`                             | `4.31.0`  |
| `torchtext`                                | `0.16.0`  |
| `bertviz`                                  | `1.4.0`   |


## Sources
•	[Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is All You Need. Advances in Neural Information Processing Systems, 30](https://arxiv.org/abs/1706.03762).
This paper introduces the Transformer architecture, which serves as the foundation for models like BERT.

•	[Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. Proceedings of NAACL-HLT 2019, 4171-4186](https://arxiv.org/abs/1810.04805).
This paper describes the BERT model used in our project for text classification tasks.

•	[Kaggle Phishing Emails Dataset](https://www.kaggle.com/katyalp/phishing-emails)


FIXME:

- sources
- report

- A short video summarizing the project can be found here - FIXME

FIXME - maybe add highlighting important passages?

