#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np 
import pandas as pd

import seaborn as sns 
import matplotlib.pyplot as plt
# Plotly Libraris
import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots

import warnings
warnings.filterwarnings("ignore")

#Users
u_cols = ['user_id', 'location', 'age']
users = pd.read_csv('data/BX-Users.csv', sep=';', names=u_cols, encoding='latin-1',low_memory=False)

#Books
i_cols = ['isbn', 'book_title' ,'book_author','year_of_publication', 'publisher', 'img_s', 'img_m', 'img_l']
items = pd.read_csv('data/BX-Books.csv', sep=';', names=i_cols, encoding='latin-1',)

#Ratings
r_cols = ['user_id', 'isbn', 'rating']
ratings = pd.read_csv('data/BX-Book-Ratings.csv', sep=';', names=r_cols, encoding='latin-1',low_memory=False)

users = users.drop(users.index[0])
items = items.drop(items.index[0])
ratings = ratings.drop(ratings.index[0])

df = pd.merge(users, ratings, on='user_id')
df = pd.merge(df, items, on='isbn')

df.head(2)

df.info()

print(df.shape)

df.isnull().sum()



# In[17]:


ds = df['rating'].value_counts().to_frame().reset_index()
ds.columns = ['value', 'count']
ds=ds.drop([0])

fig = go.Figure(go.Bar(
    y=ds['value'],x=ds['count'],orientation="h",
    marker={'color': ds['count'], 
    'colorscale': 'sunsetdark'},  
    text=ds['count'],
    textposition = "outside",
))
fig.update_layout(title_text='Rating Count',xaxis_title="Value",yaxis_title="Count",title_x=0.5)
fig.show()


# In[18]:


fig = go.Figure(go.Box(y=users['age'],name="Age")) # to get Horizonal plot change axis 
fig.update_layout(title="Distribution Of Age ",title_x=0.5)
fig.show()


# In[32]:


indexs=df[(df['rating']==0)]['user_id'].index
df_no_0=df.drop(indexs)
df_book_name=df_no_0.book_title.value_counts()[0:10].reset_index().rename(columns={'index':'book-title','book-title':'count'})


colors=['cyan','royalblue','blue','darkblue',"darkcyan",'Brown','Coral','OrangeRed','SaddleBrown','Tomato']
fig = go.Figure([go.Pie(labels=df_book_name['book_title'], values=df_book_name['count'])])
fig.update_traces(hoverinfo='label+percent', textinfo='percent+value', textfont_size=15,
                 marker=dict(colors=colors, line=dict(color='#000000', width=2)))
fig.update_layout(title="Most Reviewed Books ",title_x=0.3)
fig.show()


# In[34]:


df_book_name=df_no_0.book_title.value_counts()[0:10].reset_index().rename(columns={'index':'book-title','book-title':'count'})
df_book_name

fig = go.Figure(go.Bar(
    y=df_book_name['book_title'],x=df_book_name['count'],orientation="h",
    marker={'color': df_book_name['count'], 
    'colorscale': 'darkmint'},  
    text=df_book_name['count'],
    textposition = "outside",
))
fig.update_layout(title_text=' Top 10  Reviewed Books',xaxis_title=" Rating Count",yaxis_title="Books Name",title_x=0.6)

fig.update_layout(
    autosize=False,
    width=920,
    height=700,
   )
fig.show()


# In[20]:


import numpy as np
import pandas as pd
#import pandas_profiling
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import requests

from io import BytesIO
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly import tools
import plotly.figure_factory as ff

ratings = pd.read_csv('data/BX-Book-Ratings.csv',sep=";", encoding='latin-1')
ratings.head(10)

#  (rating data)
# Deleting duplicate columns
ratings.drop_duplicates(inplace=True, keep='first') 
print(ratings.shape)

# missing value
ratings = ratings.dropna()
print(ratings.shape)

# remove rows that rating = 0
ratings['Book-Rating'].mean()
ratings = ratings[ratings['Book-Rating'] != 0]
ratings.info()

# (books & users data)
books = pd.read_csv('data/BX-Books.csv',sep=";", encoding='latin-1')
users = pd.read_csv('data/BX-Users.csv',sep=";", encoding='latin-1')

# users setting
users_df0 = users.dropna()

# here (ratings = rating_clean)
# merging dataset
ratings.head(3)
B1 = pd.merge(ratings, users_df0, on='User-ID', how='left')
B1
B2 = pd.merge(B1, books, on='ISBN', how='left')
B2
B2.to_csv("data/B2.csv")

# cleaning
B3 = B2.dropna()
print(B3.shape)
#B3
#ratings1 = B3['Book-Rating']
#ratings1_mean = ratings1.mean() 
#ratings1_mean

# columns renaming
B3.rename(columns={
    'User-ID': 'User_ID', 
    'Book-Rating': 'Book_Rating', 
    'Book-Title': 'Book_Title',
    'Book-Author': 'Book_Author',
    'Year-Of-Publication': 'Year_Of_Publication'
}, inplace=True)
#B3.head(2)
#B3['Country'] = B3['Country'].apply(lambda x:x[:-1])
B3.head(3)
B3.to_csv("data/B3.csv")
B3.info()

# (summary statistics)
#B3.describe()
bn = B3["Book_Title"].value_counts()
bn
#B3["Book_Title"].describe()
B3["User_ID"].value_counts()
user = B3['User_ID'].astype("str")
user.describe()



# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, precision_recall_curve, precision_score
import pandas as pd

# 데이터 로드 및 전처리
data = pd.read_csv('data/B3.csv')
data['combination'] = data['Book_Title'] + " " + data['Book_Author']

# TF-IDF 벡터화
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['combination'])

# 추천 시스템 함수 (Scores 포함)
def recommendation(book_id, tfidf_matrix=tfidf_matrix, n=5):
    # 책의 index값
    idx = data.index[data['ISBN'] == book_id].to_list()[0]
    
    # 코사인 유사도 계산
    similarity = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    
    # 상위 n개의 추천 나열
    similarity_idx = similarity.argsort()[-n-1: -1][::-1]
    recommendations = data[['Book_Title', 'ISBN']].iloc[similarity_idx].copy()
    scores = similarity[similarity_idx]
    
    # Score 값에 코사인 유사도 추가
    recommendations['Score'] = scores
    
    return recommendations

# 혼동 행렬 시각화 함수
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Not Relevant', 'Relevant'], yticklabels=['Not Relevant', 'Relevant'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# Precision-Recall Curve 시각화 함수
def plot_precision_recall_curve(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()

# 성능 평가 함수에 시각화 추가
def evaluate_performance_with_visuals(book_id, data, user_id, n=10):
    # 추천된 책 목록 가져오기
    recommendations = recommendation(book_id, n=n)
    
    # 추천 목록 비어 있는지 확인
    if recommendations.empty:
        print("No recommendations found!")
        return
    
    # 사용자가 평가한 책 목록 가져오기
    user_ratings = data[data['User_ID'] == user_id]['Book_Title'].tolist()  # 사용자 ID로 가정
    
    # 추천된 책 중 사용자가 평가한 책 확인
    recommended_books = recommendations['Book_Title'].tolist()
    relevant_books = [book for book in recommended_books if book in user_ratings]
    
    # 추천 결과 출력
    print(f"Recommended books: {recommended_books}")
    print(f"Relevant books (books that user has rated): {relevant_books}")
    
    # Precision, Recall, F1 Score 계산
    y_true = [1 if book in user_ratings else 0 for book in recommended_books]
    y_pred = [1 if book in relevant_books else 0 for book in recommended_books]

    try:
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = len(relevant_books) / len(user_ratings) if len(user_ratings) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # 성능 출력
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")
        
        # 혼동 행렬 시각화
        plot_confusion_matrix(y_true, y_pred)
        
        # Precision-Recall Curve 시각화
        plot_precision_recall_curve(y_true, y_pred)
        
    except Exception as e:
        print(f"Error in calculating metrics: {e}")

# 예시: 사용자 ID 231827 성능 평가 및 시각화
book_id = '052165615X'  # 해당 ISBN으로 추천된 책 확인
user_id = 231827  # 평가할 사용자 ID

# 추천 결과 및 Scores 출력
print("Recommendations with Scores:")
print(recommendation(book_id))

# 성능 평가 및 시각화 실행
evaluate_performance_with_visuals(book_id, data, user_id)


# In[ ]:




