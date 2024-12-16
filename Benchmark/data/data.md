# Machine Learning is heading to the SUD (Socially Unacceptable Discourse) analysis: from Shallow Learning to Large Language Models to the rescue, where do we stand?
**Dimitra Niaouri, Bruno Machado Carneiro, Michele Linardi, Julien Longhi.**

## Usage

In order to reproduce the experiments conducted in this work all datasets must be acquired. All the datasets must get read and preprocessed as described in the [generate dataset](generate_dataset.py) script. The paths to the original files are hard-coded, in order to indicate exactly which of the original files were used. To generate the final dataset, all datasets are simply concatenated, as shown in the [concatenate](concatenate.py) script.  


## Data Acquisition

This work used data from the following sources:

* Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017). Automated Hate Speech Detection and the Problem of Offensive Language. In Proceedings of the 11th International AAAI Conference on Web and Social Media (pp. 512-515).  
*Source:* https://github.com/t-davidson/hate-speech-and-offensive-language  

* Founta, A.M., Djouvas, C., Chatzakou, D., Leontiadis, I., Blackburn, J., Stringhini, G., Vakali, A., Sirivianos, M., & Kourtellis, N. (2018). Large Scale Crowdsourcing and Characterization of Twitter Abusive Behavior. In 11th International Conference on Web and Social Media, ICWSM 2018.  
*Source:* http://ow.ly/BqCf30jqffN

* Lei Gao, & Ruihong Huang. (2018). Detecting Online Hate Speech Using Context Aware Models.  
*Source:* https://github.com/sjtuprog/fox-news-comments

* Grimminger, L., & Klinger, R. (2021). Hate Towards the Political Opponent: A Twitter Corpus Study of the 2020 US Elections on the Basis of Offensive Speech and Stance Detection. In Proceedings of the Eleventh Workshop on Computational Approaches to Subjectivity, Sentiment and Social Media Analysis (pp. 171–180). Association for Computational Linguistics.  
*Source:* https://www.ims.uni-stuttgart.de/data/stance_hof_us2020

* Modha, S., Mandl, T., Majumder, P., & Pate, D.. (2019). Overview of the HASOC track at FIRE 2019: Hate Speech and Offensive Content Identification in Indo-European Languages.  
*Source:* https://hasocfire.github.io/hasoc/2019/dataset.html

* Thomas Mandl, Sandip Modhab, Gautam Kishore Shahic, Amit Kumar Jaiswald, Durgesh Nandinie, Daksh Patelf, Prasenjit Majumderg, & Johannes Schäfera. (2020). Overview of the HASOC track at FIRE 2020: Hate Speech and Offensive Content Identification in Indo-European Languages.  
*Source:* https://hasocfire.github.io/hasoc/2020/dataset.html

* Basile, V., Bosco, C., Fersini, E., Nozza, D., Patti, V., Rangel Pardo, F., Rosso, P., & Sanguinetti, M. (2019). SemEval-2019 Task 5: Multilingual Detection of Hate Speech Against Immigrants and Women in Twitter. In Proceedings of the 13th International Workshop on Semantic Evaluation.  
*Source:* http://hatespeech.di.unito.it/hateval.html

* Betty van Aken, Julian Risch, Ralf Krestel, & Alexander Löser. (2018). Challenges for Toxic Comment Classification: An In-Depth Error Analysis.  
*Source:* https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data

* Zampieri, M., Malmasi, S., Nakov, P., Rosenthal, S., Farra, N., & Kumar, R. (2019). Predicting the Type and Target of Offensive Posts in Social Media. In Proceedings of NAACL.  
*Source:* https://github.com/idontflow/OLID

* Qian, J., Bethke, A., Liu, Y., Belding, E., & Wang, W. (2019). A Benchmark Dataset for Learning to Intervene in Online Hate Speech. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-ĲCNLP) (pp. 4755–4764). Association for Computational Linguistics.  
*Source:* https://github.com/jing-qian/A-Benchmark-Dataset-for-Learning-to-Intervene-in-Online-Hate-Speech

* Gibert, O., Perez, N., Garcia-Pablos, A., & Cuadros, M. (2018). Hate Speech Dataset from a White Supremacy Forum. In Proceedings of the 2nd Workshop on Abusive Language Online (ALW2) (pp. 11–20). Association for Computational Linguistics.  
*Source:* https://github.com/Vicomtech/hate-speech-dataset

* Aroyehun, A. (2018). Aggression Detection in Social Media: Using Deep Neural Networks, Data Augmentation, and Pseudo Labeling. In TRAC-2018.  
*Source:* https://github.com/kmi-linguistics/trac-1
