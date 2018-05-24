'''
Created according to this blog post: http://michelleful.github.io/code-blog/2015/06/20/pipelines/
'''

from sklearn.base import BaseEstimator, TransformerMixin

class WordCountExtractor(BaseEstimator, TransformerMixin):

  def __init__(self):
    pass

  def transform(self, X, y=None):
    print(X)
    return len(X.split())

  def fit(self, X, y=None):
      return self  # generally does nothing