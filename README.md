# A Code-First Intro to Natural Language Processing

You can find out about the course in [this blog post](https://www.fast.ai/2019/07/08/fastai-nlp/) and all [lecture videos are available here](https://www.youtube.com/playlist?list=PLtmWHNX-gukKocXQOkQjuVxglSDYWsSh9).

This course was originally taught in the [University of San Francisco's Masters of Science in Data Science](https://www.usfca.edu/arts-sciences/graduate-programs/analytics) program, summer 2019.  The course is taught in Python with Jupyter Notebooks, using libraries such as sklearn, nltk, pytorch, and fastai.

## Table of Contents
The following topics will be covered:

1\. What is NLP?
  - A changing field
  - Resources
  - Tools
  - Python libraries
  - Example applications
  - Ethics issues

2\. Topic Modeling with NMF and SVD
  - Stop words, stemming, & lemmatization
  - Term-document matrix
  - Topic Frequency-Inverse Document Frequency (TF-IDF)
  - Singular Value Decomposition (SVD)
  - Non-negative Matrix Factorization (NMF)
  - Truncated SVD, Randomized SVD

3\. Sentiment classification with Naive Bayes, Logistic regression, and ngrams
  - Sparse matrix storage
  - Counters
  - the fastai library
  - Naive Bayes
  - Logistic regression
  - Ngrams
  - Logistic regression with Naive Bayes features, with trigrams
  
4\. Regex (and re-visiting tokenization)

5\. Language modeling & sentiment classification with deep learning
  - Language model
  - Transfer learning
  - Sentiment classification

6\. Translation with RNNs
  - Review Embeddings
  - Bleu metric
  - Teacher Forcing
  - Bidirectional
  - Attention

7\. Translation with the Transformer architecture
  - Transformer Model
  - Multi-head attention
  - Masking
  - Label smoothing

8\. Bias & ethics in NLP
  - bias in word embeddings
  - types of bias
  - attention economy
  - drowning in fraudulent/fake info
  
  
 ## Why is this course taught in a weird order?

This course is structured with a *top-down* teaching method, which is different from how most math courses operate.  Typically, in a *bottom-up* approach, you first learn all the separate components you will be using, and then you gradually build them up into more complex structures.  The problems with this are that students often lose motivation, don't have a sense of the "big picture", and don't know what they'll need.

Harvard Professor David Perkins has a book, [Making Learning Whole](https://www.amazon.com/Making-Learning-Whole-Principles-Transform/dp/0470633719) in which he uses baseball as an analogy.  We don't require kids to memorize all the rules of baseball and understand all the technical details before we let them play the game.  Rather, they start playing with a just general sense of it, and then gradually learn more rules/details as time goes on.

If you took the fast.ai deep learning course, that is what we used.  You can hear more about my teaching philosophy [in this blog post](http://www.fast.ai/2016/10/08/teaching-philosophy/) or [this talk I gave at the San Francisco Machine Learning meetup](https://vimeo.com/214233053).

All that to say, don't worry if you don't understand everything at first!  You're not supposed to.  We will start using some "black boxes" and then we'll dig into the lower level details later.

To start, focus on what things DO, not what they ARE.
