import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 한글 깨짐 방지 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 1. 데이터 로드 (에러 방지용 schema_overrides 추가)
print("--- 데이터 로딩 중 ---")
movies = pl.scan_csv("movies (1).csv")
ratings = pl.scan_csv("ratings (1).csv")
# zip 컬럼을 문자열로 강제 지정하여 에러 방지
users = pl.scan_csv("users (1).csv", schema_overrides={"zip": pl.String})

# 2. 데이터 결합
df_lazy = ratings.join(movies, on="movieId", how="left") \
                 .join(users, on="userId", how="left")

# 실제 연산 수행
df_full = df_lazy.collect()
print(f"✅ 로드 완료! 전체 행 수: {df_full.height:,}개")

# ---------------------------------------------------------
# [수정된 시각화 및 EDA 코드 영역]
# ---------------------------------------------------------

# 1. 평점 분포 확인 (to_pandas()를 추가하여 호환성 해결)
print("\n1. 평점 분포 분석 중...")
rating_counts = df_full["rating"].value_counts().sort("rating").to_pandas() # Pandas로 변환

plt.figure(figsize=(8, 5))
sns.barplot(data=rating_counts, x="rating", y="count")
plt.title("전체 평점 분포")
plt.xlabel("평점")
plt.ylabel("개수")
plt.show()

# 2. 사용자별 활동성 분석
print("2. 사용자 활동성 분석 중...")
user_activity = df_full.group_by("userId").agg(
    pl.count("rating").alias("rating_count")
).sort("rating_count", descending=True)

print(f"- 평균 인당 평점 수: {user_activity['rating_count'].mean():.2f}")
print(f"- 최대 평점 수: {user_activity['rating_count'].max()}")

# 3. 성별/연령대별 평균 평점
if "gender" in df_full.columns and "age" in df_full.columns:
    print("3. 성별/연령대별 분석 중...")
    demo_stats = df_full.group_by(["gender", "age"]).agg(
        pl.col("rating").mean().alias("avg_rating")
    ).sort("age").to_pandas() # Pandas로 변환
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=demo_stats, x="age", y="avg_rating", hue="gender")
    plt.title("연령 및 성별에 따른 평균 평점 추이")
    plt.show()

# 4. 영화 인기 지표 (상위 인기작)
print("4. 인기 영화 분석 중...")
movie_stats = df_full.group_by("title").agg([
    pl.col("rating").count().alias("count"),
    pl.col("rating").mean().alias("mean")
]).filter(pl.col("count") > 500).to_pandas() # Pandas로 변환

plt.figure(figsize=(10, 6))
# 10M 전체가 아닌 요약된 데이터라 바로 그려도 됩니다.
sns.scatterplot(data=movie_stats, x="count", y="mean", alpha=0.5)
plt.title("평점 개수와 평균 평점의 관계 (상위 인기작)")
plt.xlabel("평점 개수")
plt.ylabel("평균 평점")
plt.show()

# 5. 시간 흐름에 따른 평점 변화
if "timestamp" in df_full.columns:
    print("5. 시간대별 분석 중...")
    df_time = df_full.with_columns(
        pl.from_epoch("timestamp", time_unit="s").dt.year().alias("year")
    )
    yearly_avg = df_time.group_by("year").agg(pl.col("rating").mean()).sort("year").to_pandas()
    
    plt.figure(figsize=(10, 5))
    plt.plot(yearly_avg["year"], yearly_avg["rating"], marker='o')
    plt.title("연도별 평균 평점 변화")
    plt.grid(True)
    plt.show()

print("\n--- 모든 EDA 프로세스 완료 ---")