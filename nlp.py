"""
## Overview

Unstructured data makes up the vast majority of data.  This is a basic intro to handling unstructured data.  Our objective is to be able to extract the sentiment (positive or negative) from review text.  We will do this from Yelp review data.

Your model will be assessed based on how root mean squared error of the number of stars you predict.  There is a reference solution (which should not be too hard to beat).  The reference solution has a score of 1.

**Download the data here **: http://thedataincubator.s3.amazonaws.com/coursedata/mldata/yelp_train_academic_dataset_review.json.gz


## Download and parse the data

The data is in the same format as in ml.py

## Helpful notes:
- You may run into trouble with the size of your models and Heroku's memory limit.  This is a major concern in real-world applications.  Your production environment will likely not be that different from Heroku and being able to deploy there is important and companies don't want to hire data scientists who cannot cope with this.  Think about what information the different stages of your pipeline need and how you can reduce the memory footprint.
- For example, submitting entire GridSearchCV objects will not work reliably depending on the size of your models. If you use GridSearchCV, consult the documentation to find the attributes that allow you to extract the best estimator and/or parameters. (Remember to retrain on the entire dataset before submitting.) Once again troubleshooting help can be found [here](https://sites.google.com/a/thedataincubator.com/the-data-incubator-wiki/course-information-and-logistics/course/common-miniproject-errors).
"""

'''+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'''

'''Importing modules'''
import pandas
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
import json
from sklearn.cross_validation import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin
from sklearn.linear_model import Ridge, LinearRegression, Lasso, SGDRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import pickle
import dill
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer, TfidfTransformer
import re
from sklearn.pipeline import Pipeline, FeatureUnion

'''Opening dilled txt files'''
'''Heroku paths'''
#q1_bag_of_words_model_txt = open('/app/questions/Assignment_4_NLP/nlp_Q1_bag_of_words_model.txt', 'rb+')
#q2_normalized_model_txt = open('/app/questions/Assignment_4_NLP/nlp_Q2_normalized_model.txt', 'rb+')
#q3_bigram_model_txt = open('/app/questions/Assignment_4_NLP/nlp_Q3_bigram_model.txt', 'rb+')

'''Digital Ocean paths'''
q1_bag_of_words_model_txt = open('/home/vagrant/miniprojects/questions/Assignment_4_NLP/nlp_Q1_bag_of_words_model.txt', 'rb+')
q2_normalized_model_txt = open('/home/vagrant/miniprojects/questions/Assignment_4_NLP/nlp_Q2_normalized_model.txt', 'rb+')
q3_bigram_model_txt = open('/home/vagrant/miniprojects/questions/Assignment_4_NLP/nlp_Q3_bigram_model.txt', 'rb+')

'''Loading models from dilled txt files'''
q1_bag_of_words_model = dill.load(q1_bag_of_words_model_txt)
q2_normalized_model = dill.load(q2_normalized_model_txt)
q3_bigram_model = dill.load(q3_bigram_model_txt)

'''Initializing stopwords'''
stop_words = [u'i', u'me', u'my', u'myself', u'we', u'our', u'ours', u'ourselves', u'you', u'your', u'yours', u'yourself', u'yourselves', u'he', u'him', u'his', u'himself', u'she', u'her', u'hers', u'herself', u'it', u'its', u'itself', u'they', u'them', u'their', u'theirs', u'themselves', u'what', u'which', u'who', u'whom', u'this', u'that', u'these', u'those', u'am', u'is', u'are', u'was', u'were', u'be', u'been', u'being', u'have', u'has', u'had', u'having', u'do', u'does', u'did', u'doing', u'a', u'an', u'the', u'and', u'but', u'if', u'or', u'because', u'as', u'until', u'while', u'of', u'at', u'by', u'for', u'with', u'about', u'against', u'between', u'into', u'through', u'during', u'before', u'after', u'above', u'below', u'to', u'from', u'up', u'down', u'in', u'out', u'on', u'off', u'over', u'under', u'again', u'further', u'then', u'once', u'here', u'there', u'when', u'where', u'why', u'how', u'all', u'any', u'both', u'each', u'few', u'more', u'most', u'other', u'some', u'such', u'no', u'nor', u'not', u'only', u'own', u'same', u'so', u'than', u'too', u'very', u's', u't', u'can', u'will', u'just', u'don', u'should', u'now', 'll', 'adwmuxsza', 'zu', 'abc', 'aac', 'aardbark', 'aabc', 'aab', 'aaa', 'aa', 'that', 'youre', 'zzcrkebcrfxbb', 'zse', 'i\'m', 'he\'s', 'i\'ve', 'it\'s', 'id', 'im', 'hes', 'ive', 'its']

'''Q1 MODEL - Initializing class containing Bag of Words Pipeline model'''
class q1_mlm_BOF(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self):
        self.q1_mlm_BOF = q1_bag_of_words_model
        
    def transform(self, X):
        text_list = X['text']
        a = 0
        text_list_lower_split = []
        
        text_list_lower = text_list.lower()
        text_list_split = text_list_lower.split()
        text_list_split = [w for w in text_list_split if not w in stop_words]
        text_list_split = (' '.join(text_list_split))
        text_list_split = re.findall('[a-z]{2,}', text_list_split)
        text_list_split = (' '.join(text_list_split))
        text_list_lower_split.append(text_list_split)
        
        prediction = np.asscalar(self.q1_mlm_BOF.predict(text_list_lower_split))
        return prediction
    
'''Q2 MODEL - Initializing class containing Bag of Words Normalized Pipeline model'''
class q2_mlm_BOFN(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self):
        self.q2_mlm_BOFN = q2_normalized_model
        
    def transform(self, X):
        text_list = X['text']
        a = 0
        text_list_lower_split = []
        
        text_list_lower = text_list.lower()
        text_list_split = text_list_lower.split()
        text_list_split = [w for w in text_list_split if not w in stop_words]
        text_list_split = (' '.join(text_list_split))
        text_list_split = re.findall('[a-z]{2,}', text_list_split)
        text_list_split = (' '.join(text_list_split))
        text_list_lower_split.append(text_list_split)
        
        prediction = np.asscalar(self.q2_mlm_BOFN.predict(text_list_lower_split))
        return prediction

'''Q3 MODEL - Initializing class containing Bigram Bag of Words Normalized Pipeline model'''
class q3_mlm_BBOFN(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self):
        self.q3_mlm_BBOFN = q3_bigram_model
        
    def transform(self, X):
        text_list = X['text']
        a = 0
        text_list_lower_split = []
        
        text_list_lower = text_list.lower()
        text_list_split = text_list_lower.split()
        text_list_split = [w for w in text_list_split if not w in stop_words]
        text_list_split = (' '.join(text_list_split))
        text_list_split = re.findall('[a-z]{2,}', text_list_split)
        text_list_split = (' '.join(text_list_split))
        text_list_lower_split.append(text_list_split)
        
        prediction = np.asscalar(self.q3_mlm_BBOFN.predict(text_list_lower_split))
        return prediction
    
'''*************************************************************************************************************************************'''

from lib import QuestionList, Question, list_or_dict, ListValidateMixin, YelpListOrDictValidateMixin
QuestionList.set_name("nlp")

class NLPValidateMixin(YelpListOrDictValidateMixin, Question):
    @classmethod
    def fields(cls):
        return ['text']

    @classmethod
    def _test_json(cls):
        return [
      {"votes": {"funny": 0, "useful": 0, "cool": 0}, "user_id": "WsGQfLLy3YlP_S9jBE3j1w", "review_id": "kzFlI35hkmYA_vPSsMcNoQ", "stars": 5, "date": "2012-11-03", "text": "Love it!!!!! Love it!!!!!! love it!!!!!!!   Who doesn't love Culver's!", "type": "review", "business_id": "LRKJF43s9-3jG9Lgx4zODg"},
      {"votes": {"funny": 0, "useful": 0, "cool": 0}, "user_id": "Veue6umxTpA3o1eEydowZg", "review_id": "Tfn4EfjyWInS-4ZtGAFNNw", "stars": 3, "date": "2013-12-30", "text": "Everything was great except for the burgers they are greasy and very charred compared to other stores.", "type": "review", "business_id": "LRKJF43s9-3jG9Lgx4zODg"},
      {"votes": {"funny": 0, "useful": 0, "cool": 0}, "user_id": "u5xcw6LCnnMhddoxkRIgUA", "review_id": "ZYaS2P5EmK9DANxGTV48Tw", "stars": 5, "date": "2010-12-04", "text": "I really like both Chinese restaurants in town.  This one has outstanding crab rangoon.  Love the chicken with snow peas and mushrooms and General Tso Chicken.  Food is always ready in 10 minutes which is accurate.  Good place and they give you free pop.", "type": "review", "business_id": "RgDg-k9S5YD_BaxMckifkg"},
      {"votes": {"funny": 0, "useful": 0, "cool": 0}, "user_id": "kj18hvJRPLepZPNL7ySKpg", "review_id": "uOLM0vvnFdp468ofLnszTA", "stars": 3, "date": "2011-06-02", "text": "Above average takeout with friendly staff. The sauce on the pan fried noodle is tasty. Dumplings are quite good.", "type": "review", "business_id": "RgDg-k9S5YD_BaxMckifkg"},
      {"votes": {"funny": 0, "useful": 0, "cool": 0}, "user_id": "L5kqM35IZggaPTpQJqcgwg", "review_id": "b3u1RHmZTNRc0thlFmj2oQ", "stars": 4, "date": "2012-05-28", "text": "We order from Chang Jiang often and have never been disappointed.  The menu is huge, and can accomodate anyone's taste buds.  The service is quick, usually ready in 10 minutes.", "type": "review", "business_id": "RgDg-k9S5YD_BaxMckifkg"}
    ]


@QuestionList.add
class BagOfWordsModel(NLPValidateMixin):
    """
  Build a bag of words model.  Our strategy will be to build a linear model based on the count of the words in each document (review).  **Note:** `def solution` takes an argument `record`.  Samples of `record` are given in `_test_json`.

  1. Don't forget to use tokenization!  This is important for good performance but it is also the most expensive step.  Try vectorizing as a first initial step:
    ``` python
    X = (feature_extraction.text
            .CountVectorizer()
            .fit_transform(text))
    y = scores
    ```
    and then running grid-serach and cross-validation only on of this pre-processed data.

    `CountVectorizer` has to memorize the mapping between words and the index to which it is assigned.  This is linear in the size of the focabulary.  The `HashingVectorizer` does not have to remember this mapping and will lead to much smaller models.

  2. Try choosing different values for `min_df` (minimum document frequency cutoff) and `max_df` in `CountVectorizer`.  Setting `min_df` to zero admits rare words which might only appear once in the entire corpus.  This is both prone to overfitting and makes your data unmanageablely large.  Don't forget to use cross-validation or to select the right value.  Notice that `HashingVectorizer` doesn't support `min_df`  and `max_df`.  However, it's not hard to roll your own transformer that solves for these.

  3. Try using `LinearRegression` or `RidgeCV`.  If the memory footprint is too big, try switching to Stochastic Gradient Descent: [`SGDRegressor`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html).  You might find that even ordinary linear regression fails due to the data size.  Don't forget to use `GridSearchCV` to determine the regularization parameter!  How do the regularization parameter `alpha` and the values of `min_df` and `max_df` from `CountVectorizer` change the answer?
  """
    @list_or_dict
    def solution(self, review):
        q1_mlm_BOF_class = q1_mlm_BOF()
        return q1_mlm_BOF_class.transform(review)
    
@QuestionList.add
class NormalizedModel(NLPValidateMixin):
    """
  Normalization is a key for linear regression.  Previously, we used the count as the normalization scheme.  Try some of these alternative vectorizations:

  1. You can use the "does this word present in this document" as a normalization scheme, which means the values are always 1 or 0.  So we give no additional weight to the presence of the word multiple times.

  2. Try using the log of the number of counts (or more precisely, $log(x+1)$).  This is often used because we want the repeated presence of a word to count for more but not have that effect tapper off.

  3. [TFIDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) is a common normalization scheme used in text processing.  Use the `TFIDFTransformer`.  There are options for using `idf` and taking the logarithm of `tf`.  Do these significantly affect the result?

  Finally, if you can't decide which one is better, don't forget that you can combine models with a linear regression.
  """
    @list_or_dict
    def solution(self, review):
        q2_mlm_BOFN_class = q2_mlm_BOFN()
        return q2_mlm_BOFN_class.transform(review)

@QuestionList.add
class BigramModel(NLPValidateMixin):
    """
  In a bigram model, let's consider both single words and pairs of consecutive words that appear.  This is going to be a much higher dimensional problem (large $p$) so you should be careful about overfitting.

  Sometimes, reducing the dimension can be useful.  Because we are dealing with a sparse matrix, we have to use `TruncatedSVD`.  If we reduce the dimensions, we can use a more sophisticated models than linear ones.

  As before, memory problems can crop up due to the engineering constraints. Playing with the number of features, using the HashingVectorizer, incorporating min_df and max_df limits, and handling stopwords in some way are all methods of addressing this issue. If you are using CountVectorizer, it is possible to run it with a fixed vocabulary (based on a training run, for instance). Check the documentation.

  *** A side note on multi-stage model evaluation: When your model consists of a pipeline with several stages, it can be worthwhile to evaluate which parts of the pipeline have the greatest impact on the overall accuracy (or other metric) of the model. This allows you to focus your efforts on improving the important algorithms, and leaving the rest "good enough".

  One way to accomplish this is through ceiling analysis, which can be useful when you have a training set with ground truth values at each stage. Let's say you're training a model to extract image captions from websites and return a list of names that were in the caption. Your overall accuracy at some point reaches 70%. You can try manually giving the model what you know are the correct image captions from the training set, and see how the accuracy improves (maybe up to 75%). Alternatively, giving the model the perfect name parsing for each caption increases accuracy to 90%. This indicates that the name parsing is a much more promising target for further work, and the caption extraction is a relatively smaller factor in the overall performance.

  If you don't know the right answers at different stages of the pipeline, you can still evaluate how important different parts of the model are to its performance by changing or removing certain steps while keeping everything else constant. You might try this kind of analysis to determine how important adding stopwords and stemming to your NLP model actually is, and how that importance changes with parameters like the number of features.
  """
    @list_or_dict
    def solution(self, review):
        q3_mlm_BBOFN_class = q3_mlm_BBOFN()
        return q3_mlm_BBOFN_class.transform(review)

@QuestionList.add
class FoodBigrams(ListValidateMixin, Question):
    """
  Look over all reviews of restaurants (you may need to look at the dataset from `ml.py` to figure out which ones correspond to restaurants).  There are many bigrams, but let's look at bigrams that are 'special'.  We can think of the corpus as defining an empirical distribution over all ngrams.  We can find word pairs that are unlikely to occur consecutively based on the underlying probability of their words.  Mathematically, if $p(w)$ be the probability of a word $w$ and $p(w_1 w_2)$ is the probability of the bigram $w_1 w_2$, then we want to look at word pairs $w_1 w_2$ where the statistic

  $$ p(w_1 w_2) / p(w_1) / p(w_2) $$

  is high.  Return the top 100 (mostly food) bigrams with this statistic with the 'right' prior factor (see below).

  **Questions:** (to think about: they are not a part of the answer).  This statistic is a ratio and problematic when the denominator is small.  We can fix this by applying Bayesian smoothing to $p(w)$ (i.e. mixing the empirical distribution with the uniform distribution over the vocabulary).

    1. How does changing this smoothing parameter effect the word paris you get qualitatively?

    2. We can interpret the smoothing parameter as adding a constant number of occurences of each word to our distribution.  Does this help you determine set a reasonable value for this 'prior factor'?

    3. For fun: also check out [Amazon's Statistically Improbable Phrases](http://en.wikipedia.org/wiki/Statistically_Improbable_Phrases).

  **Implementation notes:**
    - The reference solution is not an aggressive filterer. Although there are definitely artifacts in the bigrams you'll find, many of the seeming nonsense words are actually somewhat meaningful and so using smoothing parameters in the thousands or a high min_df might give you different results.
  """
    def solution(self):
        return [u'huevos rancheros'] * 100
    
'''
{u'text': "Love it!!!!! Love it!!!!!! love it!!!!!!!   Who doesn't love Culver's
!"}
---------------------------------------a----------------------------------------

{u'text': 'Everything was great except for the burgers they are greasy and very
charred compared to other stores.'}
---------------------------------------a----------------------------------------

{u'text': 'I really like both Chinese restaurants in town.  This one has outstan
ding crab rangoon.  Love the chicken with snow peas and mushrooms and General Ts
o Chicken.  Food is always ready in 10 minutes which is accurate.  Good place an
d they give you free pop.'}
---------------------------------------a----------------------------------------

{u'text': 'Above average takeout with friendly staff. The sauce on the pan fried
 noodle is tasty. Dumplings are quite good.'}
---------------------------------------a----------------------------------------

{u'text': "We order from Chang Jiang often and have never been disappointed.  Th
e menu is huge, and can accomodate anyone's taste buds.  The service is quick, u
sually ready in 10 minutes."}
---------------------------------------a----------------------------------------


{u'text': "Love it!!!!! Love it!!!!!! love it!!!!!!!   Who doesn't love Culver's
!"}
---------------------------------------b----------------------------------------

{u'text': 'Everything was great except for the burgers they are greasy and very
charred compared to other stores.'}
---------------------------------------b----------------------------------------

{u'text': 'I really like both Chinese restaurants in town.  This one has outstan
ding crab rangoon.  Love the chicken with snow peas and mushrooms and General Ts
o Chicken.  Food is always ready in 10 minutes which is accurate.  Good place an
d they give you free pop.'}
---------------------------------------b----------------------------------------

{u'text': 'Above average takeout with friendly staff. The sauce on the pan fried
 noodle is tasty. Dumplings are quite good.'}
---------------------------------------b----------------------------------------

{u'text': "We order from Chang Jiang often and have never been disappointed.  Th
e menu is huge, and can accomodate anyone's taste buds.  The service is quick, u
sually ready in 10 minutes."}
---------------------------------------b----------------------------------------


{u'text': "Love it!!!!! Love it!!!!!! love it!!!!!!!   Who doesn't love Culver's
!"}
---------------------------------------c----------------------------------------

{u'text': 'Everything was great except for the burgers they are greasy and very
charred compared to other stores.'}
---------------------------------------c----------------------------------------

{u'text': 'I really like both Chinese restaurants in town.  This one has outstan
ding crab rangoon.  Love the chicken with snow peas and mushrooms and General Ts
o Chicken.  Food is always ready in 10 minutes which is accurate.  Good place an
d they give you free pop.'}
---------------------------------------c----------------------------------------

{u'text': 'Above average takeout with friendly staff. The sauce on the pan fried
 noodle is tasty. Dumplings are quite good.'}
---------------------------------------c----------------------------------------

{u'text': "We order from Chang Jiang often and have never been disappointed.  Th
e menu is huge, and can accomodate anyone's taste buds.  The service is quick, u
sually ready in 10 minutes."}
---------------------------------------c----------------------------------------
'''