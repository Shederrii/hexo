---
title: SVM Sentiment Analysis on Movie Reviews—Report
# thumbnail:  # 略缩图
---
As I’ve been a huge movie fan, I often read people’s movie reviews to see others’ opinions on a movie—most of the times I would check if the reviews are positive or negative. However, there were times when people wrote a review that was so long and confusing that it cost me a long time to figure out the person’s attitude. As a result, I was attracted to this special programming project, where I can carry out sentiment analysis to help me figure out the attitudes inside movie reviews.

## Project Design
This movie review project employs python coding technology—such as natural language processing—into sentiment analysis, and the method used to analyze is called SVM (Support Vector Machine). It combines data science, machine learning, and linear algebra in mathematics.
- For the first step of the programming, I imported a collection of movie reviews from the nltk corpus. I used a simple and imprecise way to classify the reviews: natural language processing—identifying the positive and negative words with sentiments. By testing the frequency of such words, I was able to do a basic classification.

- Next, the SVM algorithm helps a lot to improve the standard to classify between positive and negative movie reviews. The sklearn classifier is used to classify the data in mathematic vectors.

## Python Programming
Here is the final coding process worked out for sentiment analysis, analyzing on the nltk movie review collection:
```python

import collections
import nltk.classify.util, nltk.metrics
from nltk.classify import SklearnClassifier
from nltk.metrics import precision, recall, f_measure
from nltk.corpus import movie_reviews
#2k movie reviews with sentiment polarity classification
from sklearn.svm import LinearSVC, SVC

def word_feats(words):
	return dict([(word, True) for word in words])
def cutoff(feats):
	#return int(len(feats)*19/20)
	#return int (len(feats)*(2/3))
	#return int (len(feats)*(4/5))
	#return int (len(feats)*(1/10))
	return int (len(feats)*(9/10))

negids = movie_reviews.fileids('neg')
posids = movie_reviews.fileids('pos')

negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
posfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posids]

#based on different database and negcutoff/poscutoff
#accuracy can be improved
trainfeats = negfeats[:cutoff(negfeats)] + posfeats[:cutoff(posfeats)]
testfeats = negfeats[cutoff(negfeats):] + posfeats[cutoff(posfeats):]

print('')
print('')
print('')
print('')
print('Training now on %d instances  ┌(;￣◇￣)┘!!!!!!!!!!!!!!!!'% (len(trainfeats)))
print('Test on %d instances' % len(testfeats))
print ('Using SVM algorithm~~~~~~~')
print('')
print('')
print('')
#SVM classifier
classifier = SklearnClassifier(LinearSVC(), sparse=False)
classifier.train(trainfeats)

refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)

for i, (feats, label) in enumerate(testfeats):
	refsets[label].add(i)
	observed = classifier.classify(feats)
	testsets[observed].add(i)

pos_precision = precision(refsets['pos'], testsets['pos'])
neg_precision = precision(refsets['neg'], testsets['neg'])

print ('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')		
print ('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print ('  ｡:.ﾟヽ(｡◕‿◕｡)ﾉﾟ.:｡+ﾟ     SVM    Running:          ')
print ('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print ('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


print ('SVM accuracy dprecision', (pos_precision + neg_precision) / 2)

```
(As I changed the cutoff(feats), the final accuracy result changed. )
The final results (printed ‘SVM accuracy dprecision’) are around 0.7-0.8.

## Conclusion and Evaluation
In conclusion, the SVM sentiment analysis did a good job in analyzing positive and negative emotions and had a quite high accuracy of 0.7-0.8.

This new and special method, SVM algorithm, could be a good alternative for people who want to analyze movie reviews to understand the public’s attitude quickly.

**One point of evaluation is to analyze on a more complicated movie review corpus, which would probably contain obscure emotions. However, since the SVM algorithm is designed to present an accurate standard, it should be able to handle complex situations. **
