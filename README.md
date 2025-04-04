#  COMP34812 Coursework README (Group_16)

This README describes how to run the code for our **Natural Language Understanding** coursework. Our group completed **two distinct solutions** under the **Natural Language Inference (NLI)** track:

- **Task A**: Traditional ML models (TF-IDF + Random Forest)
- **Task C**: Transformer-based fine-tuning (RoBERTa)

---

##  Project Structure

```
.
├── Demo Notebook/
│   ├── Demo_NLI_SolutionA.ipynb        # [Task A] Inference on test set using best traditional model
│   └── Demo_NLI_SolutionC.ipynb        # [Task C] Dev/test inference and prediction saving
│
├── Training Notebook/
│   ├── train_model_NLI_SolutionA.ipynb # [Task A] Traditional ML model training and evaluation
│   └── train_model_NLI_SolutionC.ipynb # [Task C] RoBERTa model training on NLI dataset
│
├── Predictions/
│   ├── Group_16_A.csv                  # [Task A] Predictions from traditional model
│   └── Group_16_C.csv                  # [Task C] Predictions from fine-tuned RoBERTa model
│
├── ModelCard/
│   ├── NLI_A_tf-idf_modelcard.md       # Model card for Task A
│   └── NLI_C_roberta_modelcard.md      # Model card for Task C
│
├── data/
│   ├── train.csv                        # Training data
│   ├── dev.csv                          # Validation data
│   ├── test.csv                         # Test data (without labels)
│   └── NLI_trial.csv                    # Trial data (for quick inspection)
│
├── model/
│   ├── SolutionA/                       # [Task A] Saved .pkl model and vectorizer, The model file is too large to be placed on Github
│   └── SolutionC/                       # [Task C] Saved RoBERTa fine-tuned model and tokenizer, The model file is too large to be placed on Github
│
└── README.md                           # This file
```

---

##  How to Run the Code

###  Task A: Traditional Model (TF-IDF + Random Forest)

1. Open `train_model_a.ipynb`  
   - Loads training/dev data  
   - Performs hyperparameter grid search  
   - Trains and evaluates Logistic Regression, Naive Bayes, Random Forest, etc.  
   - Selects best model by F1 score

2. For test inference:
   - Run `demo_code_a.ipynb`  
   - Loads saved `.pkl` model and vectorizer  
   - Generates predictions to `Group_16_A.csv`

---

###  Task C: RoBERTa Transformer Fine-Tuning

1. Open `train_model.ipynb`  
   - Loads and tokenizes the data with RoBERTa tokenizer  
   - Fine-tunes `roberta-base` over 8 epochs  
   - Best model saved to `roberta_nli_model/` based on dev F1 score

2. For prediction and evaluation:
   - Run `predict_and_eval.ipynb`  
   - Generates:
     - `predictions.csv.predict.zip` (dev) — for Codabench
     - `Group_16_C.csv` (test) — for Blackboard submission

---

##  Dependencies

```bash
pip install -r requirements.txt
```

### Key Libraries:
- For Task A: `scikit-learn`, `numpy`, `pandas`, `joblib`
- For Task C: `transformers==4.38.2`, `datasets==2.18.0`, `torch==2.2.2+cu118`, `sklearn`, `tqdm`

---

## Model Saving & Loading

### Task A
```python
# Save
joblib.dump(rf_best, 'rf_best_model.pkl')
joblib.dump(vectorizer_rf_best, 'vectorizer_rf_best.pkl')

# Load
rf_best = joblib.load('rf_best_model.pkl')
vectorizer = joblib.load('vectorizer_rf_best.pkl')
```

The trained TF-IDF + Random Forest model and vectorizer are saved to:

`model/SolutionA/`

Includes:
- `rf_best_model.pkl`
- `vectorizer_rf_best.pkl`

🔗 Model download link:  
https://drive.google.com/drive/folders/1TsKEnRT6xm9tpWEgXDUL0cHsZk9hOkdo?usp=sharing

```

### Task C
The trained RoBERTa model is saved to:

 `roberta_nli_model/`  
Includes:
- `model.safetensors`
- `config.json`
- `tokenizer_config.json`, `merges.txt`, `vocab.json`, etc.

🔗 Model download link:  
https://drive.google.com/drive/folders/1l0Kt9GoQiBsSZmFcLM6CTaIfJhA2gu_u?usp=sharing


---

##  Attribution & References

- HuggingFace Transformers: https://github.com/huggingface/transformers  
- RoBERTa model: https://huggingface.co/FacebookAI/roberta-base  
- `scikit-learn` documentation: https://scikit-learn.org  
- COMP34812 official coursework datasets

---

 **Contact Group 16**  
Developers: Weiyuan Xu, Jingyu Ji  
