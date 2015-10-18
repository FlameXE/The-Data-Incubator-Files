"""
## Overview

Our objective is to predict a new venue's popularity from information available when the venue opens.  We will do this by machine learning from a dataset of venue popularities provided by Yelp.  The dataset contains meta data about the venue (where it is located, the type of food served, etc ...).  It also contains a star rating.  This tutorial will walk you through one way to build a machine-learning algorithm.

## Metric

Your model will be assessed based on the root mean squared error of the number of stars you predict.  There is a reference solution (which should not be too hard to beat).  The reference solution has a score of 1.

## Download and parse the incoming data

[link](http://thedataincubator.s3.amazonaws.com/coursedata/mldata/yelp_train_academic_dataset_business.json.gz)

Notice that each row of the file is a json blurb.  You can read it in python.  *Hints:*
1. `gzip.open` ([docs](https://docs.python.org/2/library/gzip.html)) has the same interface as `open` but is for `.gz` files.
2. `simplejson` ([docs](http://simplejson.readthedocs.org/en/latest/)) has the same interface as `json` but is *substantially* faster.

## Setup cross-validation:
In order to track the performance of your machine-learning, you might want to use `cross_validation.train_test_split` ([docs](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.train_test_split.html)).


## Building models in sklearn

All estimators (e.g. linear regression, kmeans, etc ...) support `fit` and `predict` methods.  In fact, you can build your own by inheriting from classes in `sklearn.base` by using this template:
``` python
class Estimator(base.BaseEstimator, base.RegressorMixin):
  def __init__(self, ...):
   # initialization code

  def fit(self, X, y):
    # fit the model ...
    return self

  def predict(self, X):
    return # prediction
```
The intended usage is:
``` python
estimator = Estimator(...)  # initialize
estimator.fit(X_train, y_train)  # fit data
y_pred = estimator.predict(X_test)  # predict answer
estimator.score(X_test, y_test)  # evaluate performance
```
The regressor provides an implementation of `.score`.  Conforming to this convention has the benefit that many tools (e.g. cross-validation, grid search) rely on this interface so you can use your new estimators with the existing `sklearn` infrastructure.

For example `grid_search.GridSearchCV` ([docs](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html)) takes an estimator and some hyperparameters as arguments, and returns another estimator.  Upon fitting, it fits the best model (based on the inputted hyperparameters) and uses that for prediction.

Of course, we sometimes need to process or transform the data before we can do machine-learning on it.  `sklearn` has Transformers to help with this.  They implement this interface:
``` python
class Transformer(base.BaseEstimator, base.TransformerMixin):
  def __init__(self, ...):
   # initialization code

  def fit(self, X, y=None):
    # fit the transformation ...
    return self

  def transform(self, X):
    return ... # transformation
```
When combined with our previous `estimator`, the intended usage is
``` python
transformer = Transformer(...)  # initialize
X_trans_train = transformer.fit_transform(X_train)  # fit / transform data
estimator.fit(X_trans_train, y_train)  # fit new model on training data
X_trans_test = transformer.transform(X_test)  # transform test data
estimator.score(X_trans_test, y_test)  # fit new model
```
Here, `.fit_transform` is implemented based on the `.fit` and `.transform` methods in `base.TransformerMixin` ([docs](http://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html)).  Especially for transformers, `.fit` is often empty and only `.transform` actually does something.

The real reason we use transformers is that we can chain them together with pipelines.  For example, this
``` python
new_model = pipeline.Pipeline([
    ('trans', Transformer(...)),
    ('est', Estimator(...))
  ])
new_model.fit(X_train, y_train)
new_model.score(X_test, y_test)
```
would replace all the fitting and scoring code above.  That is, the pipeline itself is an estimator (and implements the `.fit` and `.predict` methods).  Note that a pipeline can have multiple transformers chained up but at most one (optional) terminal estimator.


## A few helpful notes about performance.

1. To deploy a model (get a trained model into Heroku), we suggest using the [`dill` library](https://pypi.python.org/pypi/dill) or [`joblib`](http://scikit-learn.org/stable/modules/model_persistence.html) to save it to disk and check it into git.  This allows you to train the model offline in another file but run it here by reading it in this file.  The model is way too complicated to be trained in real-time!

2. Make sure you load the `dill` file upon server start, not upon a call to `solution`.  This can be done by loading the model the model into the global scope.  The model is way too complicated to be even loaded in real-time!

3. Make sure you call `predict` once per call of `def solution`.  This can be done because `predict` is made to take a list of elements.

4. You probably want to use GridSearchCV to find the best hyperparameters by splitting the data into training and test.  But for the final model that you submit, don't forget to retrain on all your data (training and test) with these best parameters.

5. GridSearchCV objects are capable of prediction, but they contain many versions of your model which you'll never use. From a deployment standpoint, it makes sense to only submit the best estimator once you've trained on the full data set. To troubleshoot deployment errors look [here](https://sites.google.com/a/thedataincubator.com/the-data-incubator-wiki/course-information-and-logistics/course/common-miniproject-errors). 
"""

from lib import (QuestionList, Question, list_or_dict, catch_validate_exception,
  YelpListOrDictValidateMixin)
QuestionList.set_name("ml")

'''+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'''
'''Importing modules'''
import pandas
import numpy as np
import json
from sklearn.cross_validation import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import pickle
import dill
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline

'''Opening dilled txt files'''
'''Heroku paths'''
#q1_city_mean_model_txt = open('/app/questions/Assignment_3_ML/ml_Q1_city_mean_model.txt', 'rb+')
#q2_knn_model_txt = open('/app/questions/Assignment_3_ML/ml_Q2_knn_model.txt', 'rb+')
#q3_dict_vect_txt = open('/app/questions/Assignment_3_ML/ml_Q3_dict_vect.txt', 'rb+')
#q3_ridge_model_txt = open('/app/questions/Assignment_3_ML/ml_Q3_ridge_model.txt', 'rb+')
#q4_dict_vect_txt = open('/app/questions/Assignment_3_ML/ml_Q4_dict_vect.txt', 'rb+')
#q4_ridge_model_txt = open('/app/questions/Assignment_3_ML/ml_Q4_ridge_model.txt', 'rb+')

'''Digital Ocean paths'''
q1_city_mean_model_txt = open('/home/vagrant/miniprojects/questions/Assignment_3_ML/ml_Q1_city_mean_model.txt', 'rb+')
q2_knn_model_txt = open('/home/vagrant/miniprojects/questions/Assignment_3_ML/ml_Q2_knn_model.txt', 'rb+')
q3_dict_vect_txt = open('/home/vagrant/miniprojects/questions/Assignment_3_ML/ml_Q3_dict_vect.txt', 'rb+')
q3_ridge_model_txt = open('/home/vagrant/miniprojects/questions/Assignment_3_ML/ml_Q3_ridge_model.txt', 'rb+')
q4_dict_vect_txt = open('/home/vagrant/miniprojects/questions/Assignment_3_ML/ml_Q4_dict_vect.txt', 'rb+')
q4_ridge_model_txt = open('/home/vagrant/miniprojects/questions/Assignment_3_ML/ml_Q4_ridge_model.txt', 'rb+')

'''Loading models from dilled txt files'''
q1_city_mean_model = dill.load(q1_city_mean_model_txt)
q2_knn_model = dill.load(q2_knn_model_txt)
q3_dict_vect = dill.load(q3_dict_vect_txt)
q3_ridge_model = dill.load(q3_ridge_model_txt)
q4_dict_vect = dill.load(q4_dict_vect_txt)
q4_ridge_model = dill.load(q4_ridge_model_txt)

'''Q1 MODEL - Initializing class to compare incoming city from record with mean star rating value from city_mean_model'''
class q1_mlm_MEAN(BaseEstimator, TransformerMixin):
    def transform(self, record):
        city = record['city']
        try:
            prediction = np.asscalar(q1_city_mean_model[city])
        except:
            prediction = float(3.7)
        return prediction
    
'''Q2 MODEL - Initializing class containing a machine learning model (KNN)'''
class q2_mlm_KNN(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self):
        self.q2_mlm_KNN = q2_knn_model
        
    def transform(self, X):
        longitude_latitude = X['longitude'], X['latitude']
        prediction = np.asscalar(self.q2_mlm_KNN.predict(longitude_latitude))
        return prediction
    
'''Q3 MODEL - Initializing class containing a machine learning model (Ridge Regression)'''
class q3_mlm_RIDGE(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self):
        self.q3_mlm_RIDGE = q3_ridge_model
    
    def transform(self, X):
        category_list = []
        category_dict = {}
        dict_iter = 0
        while dict_iter < len(X['categories']):
            category_dict[str(X['categories'][dict_iter])] = str(X['categories'][dict_iter])
            dict_iter += 1
        category_list.append(category_dict)
        category_list_vect = q3_dict_vect.transform(category_list)
        prediction = np.asscalar(self.q3_mlm_RIDGE.predict(category_list_vect))
        return prediction
    
'''Q4 MODEL - Initializing class containing a machine learning model (Ridge Regression)'''
class q4_mlm_RIDGE(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self):
        self.q4_mlm_RIDGE = q4_ridge_model
    
    def transform(self, X):
        b = 0
        c = 0
        attr_list = []
        attr_dict = {}          
        while b < len(X['attributes']):
            if X['attributes'].values()[b] == str('none') or X['attributes'].values()[b] == str('no'):
                pass
            elif type(X['attributes'].values()[b]) == int:
                attr_dict[str(X['attributes'].keys()[b]) + '_' + str(X['attributes'].values()[b])] = str(X['attributes'].keys()[b]) + '_' + str(X['attributes'].values()[b])
            elif type(X['attributes'].values()[b]) == unicode:
                attr_dict[str(X['attributes'].keys()[b]) + '_' + str(X['attributes'].values()[b])] = str(X['attributes'].keys()[b]) + '_' + str(X['attributes'].values()[b])
            elif type(X['attributes'].values()[b]) == dict:
                while c < len(X['attributes'].values()[b]):
                    if X['attributes'].values()[b].values()[c] == True:
                        attr_dict[str(X['attributes'].keys()[b]) + '_' + str(X['attributes'].values()[b].keys()[c])] = str(X['attributes'].keys()[b]) + '_' + str(X['attributes'].values()[b].keys()[c])
                    c += 1
            elif X['attributes'].values()[b] == True:
                attr_dict[str(X['attributes'].keys()[b])] = str(X['attributes'].keys()[b])
            c = 0
            b += 1
        attr_list.append(attr_dict)
        b = 0
        attr_list_vect = q4_dict_vect.transform(attr_list)
        prediction = np.asscalar(self.q4_mlm_RIDGE.predict(attr_list_vect))
        return prediction
    
'''Q5 MODEL - Initializing class containing a machine learning model (Ridge Regression)'''
class q5_feature_UNION(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self):
        self.q5_feature_UNION = FeatureUnion([('q2_mlm_KNN', q2_mlm_KNN()), ('q3_mlm_RIDGE', q3_mlm_RIDGE()), ('q4_mlm_RIDGE', q4_mlm_RIDGE())])
        
    def transform(self, X):
        model_union = self.q5_feature_UNION.transform(X)
        prediction = np.asscalar(np.average(model_union))
        return prediction
        
'''*************************************************************************************************************************************'''

class MLValidateMixin(YelpListOrDictValidateMixin, Question):
    @classmethod
    def fields(cls):
        return cls._fields

    @classmethod
    def _test_json(cls):
        return [
      {"business_id": "vcNAWiLM4dR7D2nwwJ7nCA", "full_address": "4840 E Indian School Rd\nSte 101\nPhoenix, AZ 85018", "hours": {"Tuesday": {"close": "17:00", "open": "08:00"}, "Friday": {"close": "17:00", "open": "08:00"}, "Monday": {"close": "17:00", "open": "08:00"}, "Wednesday": {"close": "17:00", "open": "08:00"}, "Thursday": {"close": "17:00", "open": "08:00"}}, "open": True, "categories": ["Doctors", "Health & Medical"], "city": "Phoenix", "review_count": 7, "name": "Eric Goldberg, MD", "neighborhoods": [], "longitude": -111.98375799999999, "state": "AZ", "stars": 3.5, "latitude": 33.499313000000001, "attributes": {"By Appointment Only": True}, "type": "business"},
      {"business_id": "JwUE5GmEO-sH1FuwJgKBlQ", "full_address": "6162 US Highway 51\nDe Forest, WI 53532", "hours": {}, "open": True, "categories": ["Restaurants"], "city": "De Forest", "review_count": 26, "name": "Pine Cone Restaurant", "neighborhoods": [], "longitude": -89.335843999999994, "state": "WI", "stars": 4.0, "latitude": 43.238892999999997, "attributes": {"Take-out": True, "Good For": {"dessert": False, "latenight": False, "lunch": True, "dinner": False, "breakfast": False, "brunch": False}, "Caters": False, "Noise Level": "average", "Takes Reservations": False, "Delivery": False, "Ambience": {"romantic": False, "intimate": False, "touristy": False, "hipster": False, "divey": False, "classy": False, "trendy": False, "upscale": False, "casual": False}, "Parking": {"garage": False, "street": False, "validated": False, "lot": True, "valet": False}, "Has TV": True, "Outdoor Seating": False, "Attire": "casual", "Alcohol": "none", "Waiter Service": True, "Accepts Credit Cards": True, "Good for Kids": True, "Good For Groups": True, "Price Range": 1}, "type": "business"},
      {"business_id": "uGykseHzyS5xAMWoN6YUqA", "full_address": "505 W North St\nDe Forest, WI 53532", "hours": {"Monday": {"close": "22:00", "open": "06:00"}, "Tuesday": {"close": "22:00", "open": "06:00"}, "Friday": {"close": "22:00", "open": "06:00"}, "Wednesday": {"close": "22:00", "open": "06:00"}, "Thursday": {"close": "22:00", "open": "06:00"}, "Sunday": {"close": "21:00", "open": "06:00"}, "Saturday": {"close": "22:00", "open": "06:00"}}, "open": True, "categories": ["American (Traditional)", "Restaurants"], "city": "De Forest", "review_count": 16, "name": "Deforest Family Restaurant", "neighborhoods": [], "longitude": -89.353437, "state": "WI", "stars": 4.0, "latitude": 43.252267000000003, "attributes": {"Take-out": True, "Good For": {"dessert": False, "latenight": False, "lunch": False, "dinner": False, "breakfast": False, "brunch": True}, "Caters": False, "Noise Level": "quiet", "Takes Reservations": False, "Delivery": False, "Parking": {"garage": False, "street": False, "validated": False, "lot": True, "valet": False}, "Has TV": True, "Outdoor Seating": False, "Attire": "casual", "Ambience": {"romantic": False, "intimate": False, "touristy": False, "hipster": False, "divey": False, "classy": False, "trendy": False, "upscale": False, "casual": True}, "Waiter Service": True, "Accepts Credit Cards": True, "Good for Kids": True, "Good For Groups": True, "Price Range": 1}, "type": "business"},
      {"business_id": "LRKJF43s9-3jG9Lgx4zODg", "full_address": "4910 County Rd V\nDe Forest, WI 53532", "hours": {"Monday": {"close": "22:00", "open": "10:30"}, "Tuesday": {"close": "22:00", "open": "10:30"}, "Friday": {"close": "22:00", "open": "10:30"}, "Wednesday": {"close": "22:00", "open": "10:30"}, "Thursday": {"close": "22:00", "open": "10:30"}, "Sunday": {"close": "22:00", "open": "10:30"}, "Saturday": {"close": "22:00", "open": "10:30"}}, "open": True, "categories": ["Food", "Ice Cream & Frozen Yogurt", "Fast Food", "Restaurants"], "city": "De Forest", "review_count": 7, "name": "Culver's", "neighborhoods": [], "longitude": -89.374983, "state": "WI", "stars": 4.5, "latitude": 43.251044999999998, "attributes": {"Take-out": True, "Wi-Fi": "free", "Takes Reservations": False, "Delivery": False, "Parking": {"garage": False, "street": False, "validated": False, "lot": True, "valet": False}, "Wheelchair Accessible": True, "Attire": "casual", "Accepts Credit Cards": True, "Good For Groups": True, "Price Range": 1}, "type": "business"},
      {"business_id": "RgDg-k9S5YD_BaxMckifkg", "full_address": "631 S Main St\nDe Forest, WI 53532", "hours": {"Monday": {"close": "22:00", "open": "11:00"}, "Tuesday": {"close": "22:00", "open": "11:00"}, "Friday": {"close": "22:30", "open": "11:00"}, "Wednesday": {"close": "22:00", "open": "11:00"}, "Thursday": {"close": "22:00", "open": "11:00"}, "Sunday": {"close": "21:00", "open": "16:00"}, "Saturday": {"close": "22:30", "open": "11:00"}}, "open": True, "categories": ["Chinese", "Restaurants"], "city": "De Forest", "review_count": 3, "name": "Chang Jiang Chinese Kitchen", "neighborhoods": [], "longitude": -89.343721700000003, "state": "WI", "stars": 4.0, "latitude": 43.2408748, "attributes": {"Take-out": True, "Has TV": False, "Outdoor Seating": False, "Attire": "casual"}, "type": "business"}
    ]


@QuestionList.add
class CityModel(MLValidateMixin):
    """
  The venues belong to different cities.  You can image that the ratings in some cities are probably higher than others and use this as an estimator.

  **Note:** `def solution` takes an argument `record`.  Samples of `record` are given in `_test_json`.

  **Exercise**: Build an estimator that uses `groupby` and `mean` to compute the average rating in that city.  Use this as a predictor.
  """
    _fields = ['city']

    @list_or_dict
    def solution(self, record):
        city_mean_model_class = q1_mlm_MEAN()
        return city_mean_model_class.transform(record)


@QuestionList.add
class LatLongModel(MLValidateMixin):
    """
  You can imagine that a city-based model might not be sufficiently fine-grained.  For example, we know that some neighborhoods are trendier than others.  We might consider a K Nearest Neighbors or Random Forest based on the latitude longitude as a way to understand neighborhood dynamics.

  **Exercise**: You should implement a generic `ColumnSelectTransformer` that is passed which columns to select in the transformer and use a non-linear model like `neighbors.KNeighborsRegressor` ([docs](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)) or `ensemble.RandomForestRegressor` ([docs](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)) as the estimator (why would you choose a non-linear model?).  Bonus points if you wrap the estimator in `grid_search.GridSearchCV` and use cross-validation to determine the optimal value of the parameters.
  """
    _fields = ['longitude', 'latitude']

    @list_or_dict
    def solution(self, record):
        q2_mlm_KNN_class = q2_mlm_KNN()
        return q2_mlm_KNN_class.transform(record)


@QuestionList.add
class CategoryModel(MLValidateMixin):
    """
  Venues have categories with varying degrees of specificity, e.g.
  ```python
  [Doctors, Health & Medical]
  [Restaurants]
  [American (Traditional), Restaurants]
  ```
  With a large sparse feature set like this, we often use a cross-validated regularized linear model.
  **Exercise:**

  1. Build a custom transformer that massages the data so that it can be fed into `feature_extraction.DictVectorizer` ([docs](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html)), which in turn generates a large matrix gotten by One-Hot-Encoding.  Feed this into a Linear Regression (and cross validate it!).  Can you beat this with another type of non-linear estimator?

  1. Some categoreis (e.g. Restaurants) are not very speicifc.  Others (Japanese sushi) are much more so.  How can we account for this in our model (*Hint:* look at TF-IDF).
  """
    _fields = ['categories']

    @list_or_dict
    def solution(self, record):
        q3_mlm_RIDGE_class = q3_mlm_RIDGE()
        return q3_mlm_RIDGE_class.transform(record)


@QuestionList.add
class AttributeKnnModel(MLValidateMixin):
    """
  Venues have (potentially nested) attributes.  For example,
  ``` python
  { 'Attire': 'casual',
    'Accepts Credit Cards': True,
    'Ambience': {'casual': False, 'classy': False }}
  ```
  Categorical data like this should often be transformed by a One Hot Encoding.  For example, we might flatten the above into something like this:
  ``` python
  { 'Attire_casual' : 1,
    'Accepts Credit Cards': 1,
    'Ambience_casual': 0,
    'Ambience_classy': 0 }
  ```
  **Exercise:** Build a custom transformer that flattens attributes and feed this into `DictVectorizer`.  Feed it into a (cross-validated) linear model (or something else!)
  """
    _fields = ['attributes']

    @list_or_dict
    def solution(self, record):
        q4_mlm_RIDGE_class = q4_mlm_RIDGE()
        return q4_mlm_RIDGE_class.transform(record)


@QuestionList.add
class FullModel(MLValidateMixin):
    """
  So far we have only built models based on individual features.  We could obviously combine them.  One (highly recommended) way to do this is through a [feature union](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html).

  **Exercise:** Combine all the above models using a feature union.  Notice that a feature union takes transformers, not models as arguements.  The way around this is to build a transformer

  ```class ModelTransformer```

  that outputs the prediction in the transform method, thus turning the model into a transformer.  Use a cross-validated linear regression (or some other algorithm) to weight these signals.
  """
    _fields = None

    @list_or_dict
    def solution(self, record):
        q5_feature_UNION_class = q5_feature_UNION()
        return q5_feature_UNION_class.transform(record)

'''Closing txt files'''    
q1_city_mean_model_txt.close()
q2_knn_model_txt.close()
q3_dict_vect_txt.close()
q3_ridge_model_txt.close()
q4_dict_vect_txt.close()
q4_ridge_model_txt.close()

'''
{u'city': 'Phoenix'}
---------------------------------------a----------------------------------------
{u'city': 'De Forest'}
---------------------------------------a----------------------------------------
{u'city': 'De Forest'}
---------------------------------------a----------------------------------------
{u'city': 'De Forest'}
---------------------------------------a----------------------------------------
{u'city': 'De Forest'}
---------------------------------------a----------------------------------------


{u'latitude': 33.499313, u'longitude': -111.983758}
----------------------------------------b---------------------------------------
{u'latitude': 43.238893, u'longitude': -89.335844}
----------------------------------------b---------------------------------------
{u'latitude': 43.252267, u'longitude': -89.353437}
----------------------------------------b---------------------------------------
{u'latitude': 43.251045, u'longitude': -89.374983}
----------------------------------------b---------------------------------------
{u'latitude': 43.2408748, u'longitude': -89.3437217}
----------------------------------------b---------------------------------------


{u'categories': ['Doctors', 'Health & Medical']}
----------------------------------------c---------------------------------------
{u'categories': ['Restaurants']}
----------------------------------------c---------------------------------------
{u'categories': ['American (Traditional)', 'Restaurants']}
----------------------------------------c---------------------------------------
{u'categories': ['Food', 'Ice Cream & Frozen Yogurt', 'Fast Food', 'Restaurants'
]}
----------------------------------------c---------------------------------------
{u'categories': ['Chinese', 'Restaurants']}
----------------------------------------c---------------------------------------


{u'attributes': {'By Appointment Only': True}}
----------------------------------------d---------------------------------------
{u'attributes': {'Take-out': True, 'Price Range': 1, 'Outdoor Seating': False, '
Caters': False, 'Noise Level': 'average', 'Parking': {'garage': False, 'street':
 False, 'validated': False, 'lot': True, 'valet': False}, 'Delivery': False, 'At
tire': 'casual', 'Has TV': True, 'Good For': {'dessert': False, 'latenight': Fal
se, 'lunch': True, 'dinner': False, 'brunch': False, 'breakfast': False}, 'Takes
 Reservations': False, 'Ambience': {'hipster': False, 'romantic': False, 'divey'
: False, 'intimate': False, 'trendy': False, 'upscale': False, 'classy': False,
'touristy': False, 'casual': False}, 'Waiter Service': True, 'Accepts Credit Car
ds': True, 'Good for Kids': True, 'Good For Groups': True, 'Alcohol': 'none'}}
----------------------------------------d---------------------------------------
{u'attributes': {'Take-out': True, 'Outdoor Seating': False, 'Caters': False, 'N
oise Level': 'quiet', 'Parking': {'garage': False, 'street': False, 'validated':
 False, 'lot': True, 'valet': False}, 'Delivery': False, 'Ambience': {'hipster':
 False, 'romantic': False, 'divey': False, 'intimate': False, 'trendy': False, '
upscale': False, 'classy': False, 'touristy': False, 'casual': True}, 'Has TV':
True, 'Good For': {'dessert': False, 'latenight': False, 'lunch': False, 'dinner
': False, 'brunch': True, 'breakfast': False}, 'Takes Reservations': False, 'Att
ire': 'casual', 'Waiter Service': True, 'Accepts Credit Cards': True, 'Good for
Kids': True, 'Good For Groups': True, 'Price Range': 1}}
----------------------------------------d---------------------------------------
{u'attributes': {'Delivery': False, 'Take-out': True, 'Parking': {'garage': Fals
e, 'street': False, 'validated': False, 'lot': True, 'valet': False}, 'Price Ran
ge': 1, 'Good For Groups': True, 'Takes Reservations': False, 'Attire': 'casual'
, 'Wi-Fi': 'free', 'Accepts Credit Cards': True, 'Wheelchair Accessible': True}}

----------------------------------------d---------------------------------------
{u'attributes': {'Take-out': True, 'Has TV': False, 'Outdoor Seating': False, 'A
ttire': 'casual'}}
----------------------------------------d---------------------------------------


{u'city': 'Phoenix', u'review_count': 7, u'name': 'Eric Goldberg, MD', u'neighbo
rhoods': [], u'type': 'business', u'business_id': 'vcNAWiLM4dR7D2nwwJ7nCA', u'fu
ll_address': '4840 E Indian School Rd\nSte 101\nPhoenix, AZ 85018', u'hours': {'
Thursday': {'close': '17:00', 'open': '08:00'}, 'Tuesday': {'close': '17:00', 'o
pen': '08:00'}, 'Friday': {'close': '17:00', 'open': '08:00'}, 'Wednesday': {'cl
ose': '17:00', 'open': '08:00'}, 'Monday': {'close': '17:00', 'open': '08:00'}},
 u'state': 'AZ', u'longitude': -111.983758, u'latitude': 33.499313, u'attributes
': {'By Appointment Only': True}, u'open': True, u'categories': ['Doctors', 'Hea
lth & Medical']}
----------------------------------------e---------------------------------------
{u'city': 'De Forest', u'review_count': 26, u'name': 'Pine Cone Restaurant', u'n
eighborhoods': [], u'type': 'business', u'business_id': 'JwUE5GmEO-sH1FuwJgKBlQ'
, u'full_address': '6162 US Highway 51\nDe Forest, WI 53532', u'hours': {}, u'st
ate': 'WI', u'longitude': -89.335844, u'latitude': 43.238893, u'attributes': {'T
ake-out': True, 'Price Range': 1, 'Outdoor Seating': False, 'Caters': False, 'No
ise Level': 'average', 'Parking': {'garage': False, 'street': False, 'validated'
: False, 'lot': True, 'valet': False}, 'Delivery': False, 'Attire': 'casual', 'H
as TV': True, 'Good For': {'dessert': False, 'latenight': False, 'lunch': True,
'dinner': False, 'brunch': False, 'breakfast': False}, 'Takes Reservations': Fal
se, 'Ambience': {'hipster': False, 'romantic': False, 'divey': False, 'intimate'
: False, 'trendy': False, 'upscale': False, 'classy': False, 'touristy': False,
'casual': False}, 'Waiter Service': True, 'Accepts Credit Cards': True, 'Good fo
r Kids': True, 'Good For Groups': True, 'Alcohol': 'none'}, u'open': True, u'cat
egories': ['Restaurants']}
----------------------------------------e---------------------------------------
{u'city': 'De Forest', u'review_count': 16, u'name': 'Deforest Family Restaurant
', u'neighborhoods': [], u'type': 'business', u'business_id': 'uGykseHzyS5xAMWoN
6YUqA', u'full_address': '505 W North St\nDe Forest, WI 53532', u'hours': {'Mond
ay': {'close': '22:00', 'open': '06:00'}, 'Tuesday': {'close': '22:00', 'open':
'06:00'}, 'Friday': {'close': '22:00', 'open': '06:00'}, 'Wednesday': {'close':
'22:00', 'open': '06:00'}, 'Thursday': {'close': '22:00', 'open': '06:00'}, 'Sun
day': {'close': '21:00', 'open': '06:00'}, 'Saturday': {'close': '22:00', 'open'
: '06:00'}}, u'state': 'WI', u'longitude': -89.353437, u'latitude': 43.252267, u
'attributes': {'Take-out': True, 'Outdoor Seating': False, 'Caters': False, 'Noi
se Level': 'quiet', 'Parking': {'garage': False, 'street': False, 'validated': F
alse, 'lot': True, 'valet': False}, 'Delivery': False, 'Ambience': {'hipster': F
alse, 'romantic': False, 'divey': False, 'intimate': False, 'trendy': False, 'up
scale': False, 'classy': False, 'touristy': False, 'casual': True}, 'Has TV': Tr
ue, 'Good For': {'dessert': False, 'latenight': False, 'lunch': False, 'dinner':
 False, 'brunch': True, 'breakfast': False}, 'Takes Reservations': False, 'Attir
e': 'casual', 'Waiter Service': True, 'Accepts Credit Cards': True, 'Good for Ki
ds': True, 'Good For Groups': True, 'Price Range': 1}, u'open': True, u'categori
es': ['American (Traditional)', 'Restaurants']}
----------------------------------------e---------------------------------------
{u'city': 'De Forest', u'review_count': 7, u'name': "Culver's", u'neighborhoods'
: [], u'type': 'business', u'business_id': 'LRKJF43s9-3jG9Lgx4zODg', u'full_addr
ess': '4910 County Rd V\nDe Forest, WI 53532', u'hours': {'Monday': {'close': '2
2:00', 'open': '10:30'}, 'Tuesday': {'close': '22:00', 'open': '10:30'}, 'Friday
': {'close': '22:00', 'open': '10:30'}, 'Wednesday': {'close': '22:00', 'open':
'10:30'}, 'Thursday': {'close': '22:00', 'open': '10:30'}, 'Sunday': {'close': '
22:00', 'open': '10:30'}, 'Saturday': {'close': '22:00', 'open': '10:30'}}, u'st
ate': 'WI', u'longitude': -89.374983, u'latitude': 43.251045, u'attributes': {'D
elivery': False, 'Take-out': True, 'Parking': {'garage': False, 'street': False,
 'validated': False, 'lot': True, 'valet': False}, 'Price Range': 1, 'Good For G
roups': True, 'Takes Reservations': False, 'Attire': 'casual', 'Wi-Fi': 'free',
'Accepts Credit Cards': True, 'Wheelchair Accessible': True}, u'open': True, u'c
ategories': ['Food', 'Ice Cream & Frozen Yogurt', 'Fast Food', 'Restaurants']}
----------------------------------------e---------------------------------------
{u'city': 'De Forest', u'review_count': 3, u'name': 'Chang Jiang Chinese Kitchen
', u'neighborhoods': [], u'type': 'business', u'business_id': 'RgDg-k9S5YD_BaxMc
kifkg', u'full_address': '631 S Main St\nDe Forest, WI 53532', u'hours': {'Monda
y': {'close': '22:00', 'open': '11:00'}, 'Tuesday': {'close': '22:00', 'open': '
11:00'}, 'Friday': {'close': '22:30', 'open': '11:00'}, 'Wednesday': {'close': '
22:00', 'open': '11:00'}, 'Thursday': {'close': '22:00', 'open': '11:00'}, 'Sund
ay': {'close': '21:00', 'open': '16:00'}, 'Saturday': {'close': '22:30', 'open':
 '11:00'}}, u'state': 'WI', u'longitude': -89.3437217, u'latitude': 43.2408748,
u'attributes': {'Take-out': True, 'Has TV': False, 'Outdoor Seating': False, 'At
tire': 'casual'}, u'open': True, u'categories': ['Chinese', 'Restaurants']}
----------------------------------------e---------------------------------------
'''

