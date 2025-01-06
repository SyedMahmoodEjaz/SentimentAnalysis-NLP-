# SentimentAnalysis-NLP-

USED LIBRARIES and FUNCTIONALITIES:

•pandas: Provides dataFrames for data analysis and manipulation. 
•numpy: Supports numerical computations with arrays and matrices. 
•matplotlib.pyplot: Creates static visualizations in Python. 
•seaborn: Builds on matplotlib to provide a high-level interface for statistical graphics. 
•wordcloud: Generates visual representations of textual data. 
•nltk: Offers tools for natural language processing (NLP) tasks like tokenization, stemming, etc. 
	nltk.corpus: Provides access to various text corpora (collections of text). 
	nltk.tokenize: Implements tokenization methods (splitting text into words or sentences). nltk.stem: Offers stemming and lemmatization algorithms for word normalization. 
	nltk.tokenize.toktok: Provides a fast tokenizer for NLP tasks. 
•re: Supports regular expressions for pattern matching in strings. 
•sklearn.model_selection: Provides tools for splitting data into training and testing sets.
	sklearn.feature_extraction.text: Provides tools for text feature extraction (e.g., TF-IDF, CountVectorizer). 
	sklearn.preprocessing: Includes preprocessing techniques like label binarization. 
	sklearn.linear_model: Implements linear models like Logistic Regression. 
	sklearn.metrics: Provides metrics for evaluating machine learning models. 
	sklearn.naive_bayes: Implements Naive Bayes algorithms for classification.

Brief EXPLANATION:

This project performs sentiment analysis and word cloud generation, it also builds a LSTM and LDA model.
Data Exploration and Preprocessing:
•	Import pandas libraries for data manipulation visualization (seaborn, matplotlib), text processing (nltk), and machine learning (scikit-learn).
•	Creates a sentiment distribution plot using sns.countplot.
•	Analyzes word length distribution in positive and negative reviews.
•	Generates word clouds for positive and negative reviews using WordCloud.
•	Defines functions to remove HTML tags, special characters, and stop words from the review text.
•	Splits data into training and testing sets (not shown in the provided code).

Feature Extraction:
We have Used CountVectorizer (BOW) and TfidfVectorizer to convert text reviews into numerical features based on word frequency and importance.
Sentiment Classification Models:
Two Logistic Regression models, one for Bag-of-Words (BOW) features and another for TF-IDF features.
•	Evaluated the models performance using classification reports.

 LSTM Model:
•	LSTM model or model architecture for sentiment classification.
•	Trains the model on the preprocessed text data (likely using word embeddings).
•	It also Plots the model's training and validation accuracy.

LDA Topic Modeling:
•	LDA preprocess text for topic modeling, including stemming and removing stop words.
•	It creates a dictionary mapping words to unique IDs.
•	Converts preprocessed reviews into a bag-of-words representation.
•	Builds an LDA model to identify latent topics within the reviews.
•	Prints the top words associated with each topic.




https://colab.research.google.com/drive/1zypCQpGN3c402jx_o2OxjdKwIRHjZfRs?usp=sharing
