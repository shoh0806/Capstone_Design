import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

movies = pd.read_csv("./movies.csv")
ratings = pd.read_csv("./ratings.csv")
users = pd.read_csv("./users.csv")

# 1. 구조이해, 결측값확인, 데이터 분포보기(시각화) , 데이터 관계 분석, 이상치 확인 , 데이터 연결 , 인사이트 도출
# 1. 구조이해
print("movie info: ")
movies.info()
print("movie describe: " , movies.describe())
print(movies.head(5))
print(movies.shape)
print(movies.columns)
print("======================================================================================")
print("ratings info: ")
ratings.info()
print("ratings describe: " ,ratings.describe())
print(ratings.head(5))
print(ratings.shape)
print(ratings.columns)
print("======================================================================================")
print("users info: ")
users.info()
print("users describe: " ,users.describe())
print(users.head(5))
print(users.shape)
print(users.columns)

print("======================================================================================")
# 2. 결측값 확인 

print(movies.isna().sum())
print(ratings.isna().sum())
print(users.isna().sum())

print("======================================================================================")

# 3. 데이터 분포보기(시각화)
rating_counts = ratings['rating'].value_counts().sort_index()

rating_counts.plot(kind='bar')

plt.title('평점 분포')
plt.xlabel('평점')
plt.ylabel('개수')
plt.xticks(rotation=0)
plt.show()

print("======================================================================================")

# 4.사용자 활동성 분석 

user_activity = ratings.groupby('userId').size()

print(f"평균 평가 수: {user_activity.mean():.2f}")
print(f"최대 평가 수: {user_activity.max()}")

user_activity.hist(bins=30)

plt.title('사용자 활동성 분포')
plt.xlabel('평가 개수')
plt.ylabel('사용자 수')
plt.show()

print("======================================================================================")

# 5.연령대 평균 평점

age_map = {
    1: 'Under 18',
    18: '18-24',
    25: '25-34',
    35: '35-44',
    45: '45-49',
    50: '50-55',
    56: '56+'
}

df = pd.merge(ratings, users, on='userId')
df['age_group'] = df['age'].map(age_map)

age_rating = df.groupby('age_group')['rating'].mean()


age_rating = age_rating.reindex([
    'Under 18','18-24','25-34','35-44','45-49','50-55','56+'
])

age_rating.plot(kind='bar')

plt.title('연령대별 평균 평점')
plt.xlabel('연령대')
plt.ylabel('평균 평점')
plt.xticks(rotation=0)
plt.show()

print("======================================================================================")

# 6. 인기 영화 TOP 10

df_movie = pd.merge(ratings, movies, on='movieId')

top_movies = df_movie.groupby('title').size().sort_values(ascending=False).head(10)

top_movies.plot(kind='bar')

plt.title('인기 영화 TOP 10')
plt.xlabel('영화 제목')
plt.ylabel('평가 수')
plt.xticks(rotation=45)
plt.show()


print("인사이트: 본 분석에서는 영화 평점 데이터에 대해 탐색적 데이터 분석(EDA)을 수행하였다. 분석 결과, 전체 평점은 4점과 5점에 집중되어 사용자들이 전반적으로 긍정적인 평가를 하는 경향을 보였다. 또한 사용자 활동성은 롱테일 분포를 나타내어 일부 핵심 사용자가 많은 평가를 수행하고 있음을 확인할 수 있었다. 연령대별 분석에서는 연령에 따라 평가 성향에 차이가 존재하는 것으로 나타났으며, 영화별 평가 수 분석에서는 일부 인기 영화에 평가가 집중되는 경향이 확인되었다. 이러한 결과는 사용자 특성과 콘텐츠에 따라 평가 패턴이 달라질 수 있음을 시사한다.")
