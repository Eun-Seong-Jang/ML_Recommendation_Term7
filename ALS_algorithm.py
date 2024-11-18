import os
import implicit
import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix

class Config:
    # Dataset params
    data_path = 'data/B3.csv'
    users_count = None
    items_count = None
    val_data_size = 5
    typ = 'count'
    
    # Model params
    factors = 20
    iterations = 20
    regularization = 0.01
    show_progress = True
    
    N = 1  # Number of items to be recommended.

actions_df = pd.read_csv(Config.data_path, encoding='CP1252')

actions_df = actions_df[['User_ID', 'ISBN', 'Book_Rating']]

users = actions_df['User_ID'].unique().tolist()
items = actions_df['ISBN'].unique().tolist()

user_map = {user: idx for idx, user in enumerate(users)}
item_map = {item: idx for idx, item in enumerate(items)}

actions_df['User_ID'] = actions_df['User_ID'].map(user_map)
actions_df['ISBN'] = actions_df['ISBN'].map(item_map)

user_index = actions_df['User_ID'].values
item_index = actions_df['ISBN'].values
ratings = actions_df['Book_Rating'].values

coo = coo_matrix((ratings, (item_index, user_index)), shape=(len(items), len(users)))
csr = coo.tocsr()

model = implicit.als.AlternatingLeastSquares(factors=Config.factors, 
                                             iterations=Config.iterations, 
                                             regularization=Config.regularization, 
                                             random_state=42)

model.fit(csr)

N = 5

user_id = 276729

recommended_items = model.recommend(user_map[user_id], csr[user_map[user_id]], N=N)

recommended_books = [(items[int(recommended_item[0])], recommended_item[1]) for recommended_item in recommended_items]

print(recommended_books)
