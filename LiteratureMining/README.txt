Guide to CORD-19 Literature mining files and codes.

==========================================================================
Due to the large file size of the dataset, it will not be uploaded to LumiNUS.

Please download the dataset from:
https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge

If you do not wish to download the dataset, you may open python notebook "Cord19 mining-system output", which uses the pickled files of the dataset.

==========================================================================
Python notebooks:

1) Cord19 Literature mining step-by-step	: contains detailed code from data pre-processing to evaluation

2) Cord19 mining-system output 			: Simplified python function for system implementation

3) BioBERT fine-tuned SQuAD2 code		: Code for fine-tuning of model

==========================================================================
Model/pickle files:

1) sbert_doc		: pre-processed dataset for SBERT

2) tfidf_doc		: pre-processed dataset for TF-IDF

3) emb_corpus300	: embeddings of CORD-19 corpus
