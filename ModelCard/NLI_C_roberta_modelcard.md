---
{}
---
language: en
license: cc-by-4.0
tags:
- text-classification
repo: https://github.com/CeoXuchina/COMP34812_NLI

---

# Model Card for Weiyuan Xu-Jingyu Ji-NLI

<!-- Provide a quick summary of what the model is/does. -->

This is a natural language inference (NLI) model that classifies whether a hypothesis can be inferred from a given premise. The task is a binary classification: entailment or not.


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model is based on the RoBERTa-base transformer, fine-tuned on the COMP34812 course-provided NLI training dataset (24,432 examples) for 8 epochs. It was selected based on best F1 score on the development set.

- **Developed by:** Weiyuan Xu and Jingyu Ji
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** Transformers
- **Finetuned from model [optional]:** roberta-base

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** https://huggingface.co/FacebookAI/roberta-base
- **Paper or documentation:** https://arxiv.org/abs/1907.11692

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

24,432 natural language inference (NLI) pairs of English text, each consisting of a premise, a hypothesis, and a binary label (0 or 1). The data was provided as part of the COMP34812 coursework and used to train the model.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - learning_rate: 1e-05
      - train_batch_size: 16
      - eval_batch_size: 64
      - seed: 42
      - num_epochs: 8

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time: 1 hour 41 minutes
      - duration per training epoch: 12 minutes
      - model size: 499MB

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

A subset of the development set provided, amounting to 2K pairs.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - Precision
      - Recall
      - F1-score
      - Accuracy

### Results

The model achieved an Accuracy of 88.76% and an F1-score of 89.18% on the development set.

## Technical Specifications

### Hardware


      - RAM: at least 16 GB
      - Storage: at least 2GB,
      - GPU: Nvidia Tesla P100

### Software


      - Transformers 4.38.2
      - PyTorch 2.2.2+cu118
      - Datasets 2.18.0

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

The model may reflect biases present in the training data. Inputs longer than 512 subword tokens are truncated.It was trained only on academic domain English text.

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

The model was trained in Kaggle GPU sessions with Tesla P100. Hyperparameters were tuned based on dev set F1 score.
