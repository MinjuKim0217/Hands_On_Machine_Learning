# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## 데이터를 추출하는 함수

# +
import os
import tarfile
import urllib

DOWNLOAD_ROOT="http://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH=os.path.join("datasets","housing")
HOUSING_URL=DOWNLOAD_ROOT+"datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path=os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretreive(housing_url, tgz_path)
    housing_tgz=tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# +
import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path=os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# -

housing=pd.read_csv("housing.csv")
housing.head()

housing.info()

housing["ocean_proximity"].value_counts()

housing.describe()

# ## 데이터 훑어보기 시각화

# %matplotlib inline
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()

# ## 테스트 세트 만들기
#
# 데이터 스누핑: 전체 테스트 세트로 일반화 오차를 추정하면 매우 낙관적인 추청이 되며 시스템을 론칭했을 떄 기대한 성능이 나오지 않을 수 있다.
#
#
# 테스트 세트: 무작위로 어떤 샘플을 선택해서 데이터셋의 20% 정도를 떼어놓자

# +
import numpy as np

def split_train_test(data, test_ratio):
    shuffled_indicies=np.random.permutation(len(data))
    test_set_size=int(len(data)*test_ratio)
    test_indicies=shuffled_indicies[:test_set_size]
    train_indicies=shuffled_indicies[test_set_size:]
    
    return data.iloc[train_indicies], data.iloc[test_indicies]


# -

train_set, test_set=split_train_test(housing, 0.2)
len(train_set)

# 샘플의 식별자를 사용하여 테스트 세트로 보낼지 말지 결정해보자
#
# 새로운 테스트 세트는 새 샘플의 20%를 갖게 되지만 이전에 훈련 세트에 있던 샘플은 포함시키지 않을 것이다. 

# +
from zlib import crc32

def test_set_check(identifier,test_ratio):
    return crc32(np.int64(identifier))&0xffffffff<test_ratio*2*32

def split_train_test_by_id(data, test_ratio, id_column):
    ids=data[id_column]
    in_test_set=ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


# +
housing_with_id=housing.reset_index() #'index' 열이 추가된 데이터프레임이 반환된다. 

train_set, test_set=split_train_test_by_id(housing_with_id,0.2,"index")
# -

# 고유 식별자를 만드는 데 안전한 특성을 사용해야 합니다.
# ex) 경도와 위도를 합한 값을 아이디로 둘 수 있다.

housing_with_id["id"]=housing["longitude"]*1000+housing["latitude"]
train_set, test_set=split_train_test_by_id(housing_with_id, 0.2, "id")

# +
from sklearn.model_selection import train_test_split

train_set, test_set=train_test_split(housing, test_size=0.2, random_state=42)
# -

# pd.cut() 함수를 사용해 카테고리 5개를 가진 소득 카테고리 특성을 만들에 낸다 
#
# 왜냐하면 계층별로 데이터셋에 충분한 샘플 수가 있어야 하고 만약 그렇지 않는다면 계층의 중요도를 추정하는 데 편향이 발생하기 때문이다.

housing["income_cat"]=pd.cut(housing["median_income"],
                             bins=[0.,1.5,3.0,4.5,6.,np.inf],
                             labels=[1,2,3,4,5])

housing["income_cat"].hist()

# 소득 카테고리 별로 계층샘플링을 해보자

# +
from sklearn.model_selection import StratifiedShuffleSplit

split=StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set=housing.loc[train_index]
    strat_test_set=housing.loc[test_index]
# -

strat_test_set["income_cat"].value_counts()/len(strat_test_set)
# 테스트 세트에서 소득 카테고리의 비율

# +
#income_cat 특성을 삭제해서 데이터를 원래 상태로 되돌려 놓기

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
# -

# # 데이터 이해를 위한 탐색과 시각화

# ## 지리적 데이터 시각화

# +
# 훈련세트 손상시키지 않게 하기 위해 복사본을 만들자

housing=strat_train_set.copy()

# +
#모든 구역을 산점도로 만들어 데이터를 시각화 해보자

housing.plot(kind="scatter", x="longitude", y="latitude")

# +
# alpha를 0.1로 주어 데이터 포인트가 밀집된 영역을 나타내보자

housing.plot(kind="scatter", x='longitude', y="latitude", alpha=0.1)

# +
#주택 가격 나타내기

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing["population"]/100, label="population", figsize=(10,7), c="median_house_value", 
             cmap=plt.get_cmap("jet"), colorbar=True, sharex=False)
plt.legend()
# -

# 캘리포니아주택 가격: 빨간색은 높은 가격, 파란색은 낮은 가격, 큰 원은 인구가 밀집된 지역을 나타낸다.

# ## 상관관계 조사
#
# 데이터셋이 너무 크지 않으므로 모든 특성간의 표준 상관계수를 corr() 메서드를 이용해 쉽게 계산할 수 있다,

corr_matrix=housing.corr()

corr_matrix['median_house_value'].sort_values(ascending=True)
# 값이 1에 가까울 수록 강한 양의 상관관계를 가진다 

# +
# 판다스로 상관관계 알아보기

from pandas.plotting import scatter_matrix

attributes=["median_house_value", "median_income","total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12,8))
# -

housing.plot(kind='scatter', x='median_income',y='median_house_value', alpha=0.1)

# 위의 그래프: 1) 상관관계가 매우 강하다. 포인트들이 너무 널리 퍼져있지 않다, 
#
# 2)앞서 본 가격 제한값이 $500,00에서 수평선으로 잘 보이낟. $450,000부근에서도 잘 보이고 $350,000 부근에서보 관측이 된다. 
#
# 알고리즘이 데이터에서 이런 이상한 형태를 학습하지 않도록 제거를 해주어야 한다. 

# ### 특성조합으로 실험
#
# 여러가지 조합으로 유용한 데이터셋을 만들어보자

housing["rooms_per_household"]=housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"]=housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

corr_matrix=housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=True)

# ## 머신러닝 알고리즘을 위한 데이터 준비

# +
#원래의 훈련세트로 되돌아오자
# 예측 변수와 타깃값에 같은 변형을 적용하지 않기 위해 예측 변수와 레이블을 분리해주자

housing=strat_train_set.drop("median_house_value", axis=1)
housing_labels=strat_train_set["median_house_value"].copy()
# -

# ### 데이터 정제

# +
# total_bedroom에 특성값 없는 경우를 없애주자

housing.dropna(subset=["total_bedrooms"])
#housing.drop("total_bedrooms",axis=1)
#median=housing["total_bedrooms"].median()

# +
# 누락된 값을 손쉽게 다루게 해주는 SimpleInputer

from sklearn.impute import SimpleImputer

imputer=SimpleImputer(strategy="median")

# +
#중간값이 수치형 특성에서만 계산될 수 있기 때문에 텍스트 특성인 ocean_proximity를 제외한 데이터 복사본 생성

housing_num=housing.drop("ocean_proximity", axis=1)
# -

imputer.fit(housing_num)

imputer.statistics_

housing_num.median().values

X=imputer.transform(housing_num)

housing_tr=pd.DataFrame(X, columns=housing_num.columns,index=housing_num.index)

# ### 텍스트와 범주형 특성 다루기

housing_cat=housing[["ocean_proximity"]]
housing_cat.head(10)

from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder=OrdinalEncoder()
housing_cat_encoded=ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]

# 범주형 특성마다 카테고리들의 1D 배열을 담은 리스트 반환
ordinal_encoder.categories_

# +
#one=hot encoding

from sklearn.preprocessing import OneHotEncoder
cat_encoder=OneHotEncoder()
housing_cat_1hot=cat_encoder.fit_transform(housing_cat)
housing_cat_1hot
# -

#사이파이 희소행렬을 numpy 배열로 변경 toarray()
housing_cat_1hot.toarray()

cat_encoder.categories_

# ## 나만의 변환기 만들기

# +
# 조합특성을 추가하는 간단한 변환기

from sklearn.base import BaseEstimator, TransformerMixin

# get the right column indices: safer than hard-coding indices 3, 4, 5, 6
rooms_ix, bedrooms_ix, population_ix, household_ix = [
    list(housing.columns).index(col)
    for col in ("total_rooms", "total_bedrooms", "population", "households")]

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kwargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs=attr_adder.transform(housing.values)
# -

# ### 특성 스케일링, 변환 파이프 라인

# +
# 연속된 변환을 순서대로 처리할 수 있도록 도와주는 Pipeline 클래스 존재한다. 
# 숫자 특성 처리하는 간단한 파이프라인

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline=Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

housing_num_tr=num_pipeline.fit_transform(housing_num)

# +
# 하나의 변환기로 적절하게 모든 열 변환 수행 ColumnTransformer

from sklearn.compose import ColumnTransformer

num_attribs=list(housing_num)
cat_attribs=["ocean_proximity"]

full_pipeline=ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

housing_prepared=full_pipeline.fit_transform(housing)
# -

# ## 모델 선택과 훈련

# ### 훈련 세트에서 훈련하고 평가하기

# +
# 선형 회귀 모델 훈련

from sklearn.linear_model import LinearRegression

lin_reg=LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# +
some_data=housing.iloc[:5]
some_labels=housing_labels.iloc[:5]
some_data_prepared=full_pipeline.transform(some_data)

print("예측", lin_reg.predict(some_data_prepared))
# -

print("레이블", list(some_labels))

# +
# mean_squared_error 를 통해서 전체 훈련 세트에 대해 이 회귀 모델의 RMSE 측정

from sklearn.metrics import mean_squared_error

housing_predictions=lin_reg.predict(housing_prepared)
lin_mse=mean_squared_error(housing_labels, housing_predictions)
lin_rmse=np.sqrt(lin_mse)
lin_rmse

# +
# DecisionTreeRegrssion,  데이터에서 복잡한 비선형 관계를 찾을 수 있다.

from sklearn.tree import DecisionTreeRegressor

tree_reg=DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

# +
#훈령세트로 평가

housing_predictions=tree_reg.predict(housing_prepared)
tree_mse=mean_squared_error(housing_labels, housing_predictions)
tree_rmse=np.sqrt(tree_mse)
tree_rmse
# -

# ### 교차 검증을 사용한 평가
#

# +
# k fold cross validation

from sklearn.model_selection import cross_val_score

scores=cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores=np.sqrt(-scores)


# -

def display_scores(scores):
    print("점수:", scores)
    print("평균:", scores.mean())
    print("표준편차", scores.std())
display_scores(tree_rmse_scores)


# 결정 트리 결과는 선형회귀 모델보다 결과가 나쁘다는 것을 알 수 있다. 

# 선형 회귀 모델의 점수
lin_scores=cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores=np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

# +
#Random Forest Regressor-> 특성을 무작위로 선택해서 많은 결정 트리를 만들고그 예측을 평균내는 방식으로 작동
from sklearn.ensemble import RandomForestRegressor

forest_reg=RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
forest_reg

# -

display_scores(forest_rmse_scores)

# ## 모델 세부 튜닝

# ### 그리드 탐색
#

# 수동으로 하이퍼 파라미터 조정해보자. GridSearchCV를 사용해서 탐색하고자 하는 하이퍼파라미터와 시도해볼 값을 지정해주자
#

# +
from sklearn.model_selection import GridSearchCV

param_grid = [
    # 12(=3 X 4)개의 하이퍼 파라미터 조합을 시도합니다.
    {'n_estimators':[3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # bootstrap은 false로 하고 6(=2 X 3)개의 조합을 시도합니다.
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

forest_reg = RandomForestRegressor(random_state=42)
# 다섯 개의 폴드로 훈련하면 총 (12+6)*5=90번의 훈련이 일어납니다.

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                          scoring='neg_mean_squared_error',
                          return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)
# -

#최적의 파라미터
grid_search.best_params_

#최적의 추정기
grid_search.best_estimator_

cvres=grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


