# Machine_Learning_heading_to_SUD
**Dimitra Niaouri, Bruno Machado Carneiro, Michele Linardi, Julien Longhi**

## Data
We selected 13 public datasets originating from different annotation schemas, totaling 470,768 samples across 12 classes. 

| Dataset           | Source                            | Sample Type                  | # Samples     | Topic                         |
|-------------------|-----------------------------------|------------------------------|---------------|------------------------------------|
| **Davidson**      | Davidson et al., 2017            | Tweets                       | 25,000        | Generic                           |
| **Founta**        | Founta et al., 2018              | Tweets                       | 100,000       | Generic                           |
| **Fox**           | Yuan and Rizoiu, 2022            | Threads                      | 1,528         | Fox News Posts                    |
| **Gab**           | Qian et al., 2019                | Posts                        | 34,000        | Generic                           |
| **Grimminger**    | Grimminger and Klinger, 2021     | Tweets                       | 3,000         | US Presidential Election          |
| **HASOC2019**     | Wang et al., 2019                | Facebook, Twitter posts      | 12,000        | Generic                           |
| **HASOC2020**     | Ghosh Roy et al., 2021           | Facebook posts               | 12,000        | Generic                           |
| **Hateval**       | MacAvaney et al., 2019           | Tweets                       | 13,000        | Misogynist and Racist content     |
| **Jigsaw**        | van Aken et al., 2018            | Wikipedia talk pages         | 220,000       | Generic                           |
| **Olid**          | Zampieri et al., 2019            | Tweets                       | 14,000        | Generic                           |
| **Reddit**        | Yuan and Rizoiu, 2022            | Posts                        | 22,000        | Toxic subjects                    |
| **Stormfront**    | MacAvaney et al., 2019           | Threads                      | 10,500        | White Supremacy Forum             |
| **Trac**          | Aroyehun and Gelbukh, 2018       | Facebook posts               | 15,000        | Generic                           |

*Table 1: Summary of datasets (Carneiro et al., 2023)*


Details on the data acquisition and preprocessing are described [here](data/data.md).


## Models
We designed a framework of SOTA models differentiating between 3 model families: Shallow Learning Models (SLMs), Masked Language Models (MLMs), and Causal Language Models (CLMs). A summary of the models we finetuned can be found below:


| Category | Models                          | Citation                      |
|----------|---------------------------------|-------------------------------|
| **SLM**  | **(Shallow Learning Models)**   |                               |
|          | Stochastic Gradient Boosting (SGB) | Friedman (2001)               |
|          | Logistic Regression (LR)        | Wright (1995)                 |
|          | Multinomial Naive Bayes (MNB)   | Kibriya et al. (2003)         |
|          | Random Forest (RF)              | Breiman (2001)                |
|          | Support Vector Machines (SVM)   | Hearst et al. (1998)          |
| **MLM**  | **(Masked Language Models)**    |                               |
|          | BERTBASE                        | Devlin et al. (2019)          |
|          | ALBERTBASE                      | Lan et al. (2019)             |
|          | RoBERTaBASE                     | Liu et al. (2019)             |
|          | ELECTRABASE                     | Clark et al. (2020)           |
| **CLM**  | **(Causal Language Models)**    |                               |
|          | Llama-2-7b-hf                   | Touvron et al. (2023)         |
|          | Mistral-7B-v0.1                 | Jiang et al. (2023)           |
|          | mpt-7b                          | MosaicML NLP Team (2023)      |

* **Shallow Learning Models**   

Shallow learning models, defined as a category encompassing traditional ML algorithms proposed before 2006, are characterized by their simplicity, typically featuring few layers or processing units (Xu et al., 2021). These models are well-suited for tasks with straightforward data patterns. However, their basic architecture may limit their capacity to capture complex relationships and adapt to new data. Consequently, the performance of such models heavily relies on the efficacy of the feature extraction process (Janiesch et al., 2021). Within this overarching classification, we specifically explore Gradient Boosting (GB), Logistic Regression (LR), Multinomial Naive Bayes (MNB), Random Forest (RF), and Support Vector Machines (SVM). 

* **Masked Language models**  

Masked language models (MLMs), as described in Devlin et al. (2019), are deep learning models that have been trained to fill in the blanks for masked tokens in a given input sequence. Specifically, MLMs aim to predict the original vocabulary identity of a masked word, relying solely on the context provided by surrounding words. The key advantage of these models is their ability to consider both preceding and subsequent tokens in the input sequence, enabling a bidirectional understanding during the prediction process. Masked Language models are acclaimed for their high performances in classification tasks. Within this category, we finetune and assess the performance of BERTBASE (Devlin et al., 2019; Yuan and Rizoiu, 2022) and some of its architectural variants introduced to enhance overall performance and reduce computational complexity, namely ALBERTBASE, (Lan et al., 2019)  RoBERTaBASE (Liu et al., 2019) and ELECTRABASE (Clark et al., 2020). 

* **Causal Language models**  

As explained in the previous section, MLMs are bidirectional models trained to comprehend context from both directions. In contrast, CLMs, are unidirectional models that only consider the preceding context for predictions. CLMs are trained to anticipate the next token in a sequence solely based on prior tokens, making them particularly adept at text generation tasks. The CLM models fine-tuned and evaluated in this study are Llama 2 (Llama-2-7b-hf) (Touvron et al. (2023), Mistral (Mistral-7B-v0.1) (Jiang et al., 2023) and MPT (mpt-7b) (MosaicML NLP team, 2023). 
## Main Results

* **Optimal Performing Model Per Class and Dataset**

| Dataset     | Abusive | Aggressive | Hate | Identity Hate | Insult | Neither | Obscene | Offensive | Profane | Severe Toxic | Threat | Toxic | Best Model |
|-------------|---------|------------|------|---------------|--------|---------|---------|-----------|---------|--------------|--------|-------|------------|
| **GSUD**    | 0.79    | 0.64       | 0.67 | 0.6           | 0.68   | 0.36    | 0.42    | 0.5       | 0.94    | 0.25         | 0.75   | 0.31  | BERT       |
|             | 0.8     | 0.64       | 0.6  | 0.38          | 0.51   | 0.94    | 0.34    | 0.75      | 0.94    | 0.34         | 0.75   | 0.33  | ELECTRA    |
|             | 0.8     | 0.67       | 0.68 | 0.42          | 0.5    | 0.94    | 0.25    | 0.75      | 0.37    | 0.42         | 0.46   | 0.17  | RoBERTa    |
| **Davidson**| -       | -          | 0.46 | -             | 0.9    | -       | 0.94    | -         | -       | -            | -      | -     | ELECTRA    |
| **Founta**  | 0.89    | -          | 0.42 | -             | -      | 0.91    | -       | -         | -       | -            | -      | -     | MISTRAL    |
| **Fox**     | -       | -          | 0.67 | -             | -      | 0.82    | -       | -         | -       | -            | -      | -     | MISTRAL    |
| **Gab**     | -       | -          | -    | -             | 0.89   | -       | -       | -         | 0.91    | -            | -      | -     | GB         |
|             |         |            |      |               | 0.88   |         |         |           | 0.91    |              |        |       | ALBERT     |
|             |         |            |      |               | 0.89   |         |         |           | 0.91    |              |        |       | RoBERTa    |
| **Grimminger** | -    | -          | 0.58 | -             | -      | 0.95    | -       | -         | -       | -            | -      | -     | ELECTRA    |
| **HASOC2019** | -     | -          | 0.29 | -             | -      | 0.8     | -       | 0.36      | 0.57    | -            | -      | -     | ELECTRA    |
| **HASOC2020** | -     | -          | 0.22 | -             | -      | 0.91    | -       | 0.3       | 0.83    | -            | -      | -     | ELECTRA    |
| **Hateval** | -       | -          | -    | -             | 0.75   | -       | -       | -         | 0.79    | -            | -      | -     | ELECTRA    |
|             |         |            |      |               | 0.75   |         |         |           | 0.8     |              |        |       | RoBERTa    |
|             |         |            |      |               | 0.76   |         |         |           | 0.78    |              |        |       | MISTRAL    |
| **Jigsaw**  | -       | -          | -    | 0.46          | 0.57   | 0.98    | 0.38    | -         | -       | 0.4          | 0.56   | 0.3   | ELECTRA    |
| **Olid**    | -       | -          | -    | -             | -      | -       | -       | 0.85      | -       | 0.67         | -      | -     | BERT       |
|             |         |            |      |               |        |         |         | 0.84      |         | 0.68         |        |       | ELECTRA    |
| **Reddit**  | -       | -          | -    | -             | 0.77   | -       | -       | 0.92      | -       | -            | -      | -     | LLAMA 2    |
|             |         |            |      |               | 0.78   |         |         | 0.93      |         |              |        |       | MISTRAL    |
| **Stormfront** | -    | -          | 0.6  | -             | -      | 0.96    | -       | -         | -       | -            | -      | -     | RoBERTa    |
| **Trac**    | -       | 0.84       | -    | -             | -      | 0.71    | -       | -         | -       | -            | -      | -     | MISTRAL    |

## Dependencies  

* **Install all the dependencies:**  
```
python3 -m pip install -r requirements.txt
```
## Repository Structure
```
Machine_Learning_heading_to_SUD
├── data
    │   ├── concatenate.py
    │   ├── generate_dataset.py
    │   └── preprocessing.py
└── src
    ├── CLMs
    │   ├── Llama2.py
    │   ├── Mistral.py
    │   └── MPT.py
    ├── MLMs
    │   ├── BERT_models.py
    │   ├── BERT_tf_hub.json
    │   ├── load_data.py
    │   ├── test.py
    │   └── train.py
    └── SLMs
        ├── GB.py
        ├── LR.py
        ├── MNB.py
        ├── RF.py
        └── SVM.py
```

## Usage  

### *For CLMs*
* **Arguments:**
```
--data          Path to the dataset
```

### *For MLMs*

### Training  

```
train.py [--data data_path] 
         [--mode {binary, multi-class}]
         [--batch_size batch_size]
         [--epochs epochs_count]
         [--lr learning_rate]
         [--smoothing smoothing_rate]
```

* **Arguments:**
```
--data          Path to the dataset
--mode          Perform binary or multi-class SUD classification
--batch_size    Batch size
--epochs        Number of epochs
--lr            Learning rate
--smoothing     Label smoothing rate
```

### Testing  

```
test.py [--data data_path] 
        [--model model_path]
        [--mode {binary, multi-class}]
```

* **Arguments:**
```
--data          Path to the dataset
--model         Path to the trained model
--mode          Perform binary or multi-class SUD classification
```

### *For SLMs*
* **Arguments:**
```
--data          Path to the dataset
```
      
## References

* Breiman, L. Random Forests. Machine Learning 45, 5–32 (2001). https://doi.org/10.1023/A:1010933404324.

* Clark, K., Luong, M. T., Le, Q. V., & Manning, C. D. (2020). Electra: Pre-training text encoders as discriminators rather than generators.

* Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. (2019). Bert: Pre-training of deep bidirectional transformers for language understanding.

* Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. Annals of statistics, 1189-1232.

* Hearst, M. A., Dumais, S. T., Osuna, E., Platt, J., & Scholkopf, B. (1998). Support vector machines. IEEE Intelligent Systems and their applications, 13(4), 18-28.
  
* Janiesch, C., Zschech, P. & Heinrich, K. Machine learning and deep learning. Electron Markets 31, 685–695 (2021). https://doi.org/10.1007/s12525-021-00475-2. 

* Jiang, A., Sablayrolles, A., Mensch, A., Bamford, C., Devendra, S., Chaplot, D., De Las Casas, F., Bressand, G., Lengyel, G., Lample, L., Saulnier, R., Lavaud, M.-A., Lachaux, P., Stock, T., Le Scao, T., Lavril, T., Wang, T., Lacroix, W., & Sayed. (2023). Mistral 7B. 

* Kibriya, A. M., Frank, E., Pfahringer, B., & Holmes, G. (2005). Multinomial naive bayes for text categorization revisited. In AI 2004: Advances in Artificial Intelligence: 17th Australian Joint Conference on Artificial Intelligence, Cairns, Australia, December 4-6, 2004. Proceedings 17 (pp. 488-499). Springer Berlin Heidelberg.

* Lan, Z., Chen, M., Goodman, S., Gimpel, K., Sharma, P., & Soricut, R. (2019). Albert: A lite bert for self-supervised learning of language representations. arXiv preprint arXiv:1909.11942. 

* Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., Stoyanov, V., & Allen, P. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. 

* MosaicML NLP Team. (2023). Introducing MPT-7B: A New Standard for Open-Source, Commercially Usable LLMs. Databricks. https://www.databricks.com/blog/mpt-7b.

* Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., Rodriguez, A., Joulin, A., Grave, E., & Lample, G. (2023). LLaMA: Open and Efficient Foundation Language Models.

* Wright, R. E. (1995). Logistic regression. In L. G. Grimm & P. R. Yarnold (Eds.), Reading and understanding multivariate statistics (pp. 217–244). American Psychological Association.
  
* Yuan, L. and Rizoiu, M.-A. (2022). Detect hate speech in unseen domains using multi-task learning: A case study of political public figures.
