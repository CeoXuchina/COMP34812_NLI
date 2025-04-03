#  COMP34812 Coursework README (Group_16)

This README describes how to run the code for our **Natural Language Understanding** coursework. Our group completed **two distinct solutions** under the **Natural Language Inference (NLI)** track:

- **Task A**: Traditional ML models (TF-IDF + Random Forest)
- **Task C**: Transformer-based fine-tuning (RoBERTa)

---

##  Project Structure

```
.
├── train_model_a.ipynb             # [Task A] Traditional ML model training and evaluation
├── demo_code_a.ipynb               # [Task A] Inference on test set using best traditional model
├── Group_16_A.csv                  # [Task A] Predictions from traditional model

├── train_model.ipynb               # [Task C] RoBERTa model training on NLI dataset
├── predict_and_eval.ipynb         # [Task C] Dev/test inference and prediction saving
├── Group_16_C.csv                 # [Task C] Predictions from fine-tuned RoBERTa model
├── predictions.csv.predict.zip    # [Task C] Dev set predictions for Codabench

├── Group_16_A_model_card.md        # Model card for Task A
├── Group_16_C_model_card.md        # Model card for Task C
├── poster.pdf                      # Flash poster for presentation
└── README.md                       # This file
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
