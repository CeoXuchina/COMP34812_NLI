---
{}
---
language: en
license: cc-by-4.0
tags:
- text-classification
repo: https://github.com/CeoXuchina/COMP34812_NLI

---

# Model Card for m75875jj-x69739wx-NLI

<!-- Provide a quick summary of what the model is/does. -->

This is a classification model that was trained to determine the relationship between a pair of text sequences in the Natural Language Inference (NLI) task.


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model is based on a TF-IDF vectorizer combined with a Random Forest classifier. It was selected as the best-performing model after training and evaluating multiple traditional
    machine learning models (including Logistic Regression, Naive Bayes, and SVM) across several hyperparameter combinations.

- **Developed by:** Jingyu Ji and Weiyuan Xu
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** TF-IDF + Random Forest
- **Finetuned from model [optional]:** None

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** None
- **Paper or documentation:** https://link.springer.com/article/10.1023/A:1010933404324

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

24K+ premise-hypothesis pairs as training data

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - max_features: 5000
      - ngram_range: (1,3)
      - seed: 42
      - min_df: 2
      - max_df: 0.9
      - n_estimators: 100

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time: ～30 minutes
      - duration per training epoch: ~30 minutes
      - model size: 108MB

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

A subset of the development set provided, amounting to 6K pairs.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - F1-score: 0.627
      - Accuracy: 0.624
      

### Results

The model obtained an F1-score of 63% and an accuracy of 63%.

## Technical Specifications

### Hardware


      - RAM: at least 16 GB
      - Storage: at least 2GB,
      - GPU: T4x2

### Software


      - scikit-learn 1.3.0
      - pandas 1.5.3
      - numpy 1.24.2
      - joblib 1.2.0
      

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

As a traditional machine learning model based on TF-IDF, this model does not consider word order or deep contextual semantics. It may be sensitive to vocabulary distribution and fail to generalize on out-of-distribution or low-resource domain inputs.

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

The hyperparameters were selected through a manual grid search over TF-IDF settings and model-specific parameter combinations. The final configuration was chosen based on validation set performance (weighted F1 score).
    Also, the text might not be preprocessed very well, only case folding and keeping English words, numbers and space.  
