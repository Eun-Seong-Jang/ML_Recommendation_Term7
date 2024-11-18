from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# 데이터 불러오기
data = pd.read_csv("C:\\Users\\admin\\Desktop\\가천대\\3학년 2학기\\머신러닝\\TermProject\\B3.csv")

# 텍스트 유사도 계산을 위해 제목과 저자 결합
data['combination'] = data['Book_Title'] + " " + data['Book_Author']

# TF-IDF matrix 생성
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['combination'])

# 추천 함수
def recommendation(book_id, tfidf_matrix=tfidf_matrix, n=5):
    # 책의 index값
    idx = data.index[data['ISBN'] == book_id].to_list()[0]
    
    # 코사인 유사도 계산
    similarity = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    
    # 상위 n개의 추천 나열
    similarity_idx = similarity.argsort()[-n-1: -1][::-1]
    recommendations = data[['Book_Title', 'ISBN']].iloc[similarity_idx].copy()
    scores = similarity[similarity_idx]
    
    # Score 값에 코사인 유사도 넣기
    recommendations['Score'] = scores
    
    return recommendations

# 테스트
book_id = '052165615X'
print(recommendation(book_id))