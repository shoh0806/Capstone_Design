import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 일단 pip install 해서 다 설치한 후에 pandas, numpy, sklearn tensorflow를 import해 왔습니다. 
# 진행하다가 만약 더 필요한게 있다면 추가하면 될거같고요 아래 datetime은 아무래도 978300019 이런식으로 초를
# 보는것보다는 몇년 몇월 며칠 몇시 몇분 몇초로 나타내는게 나을거같아서 아래에서 import해서 사용했습니다.
# 승환님이 돌리실떄는 아래 csv파일 경로만 승환님 경로로 바꿔서 진행해보시면 될거같아요. 
# 그리고 .head같이 확인해보는데 쓴것들은 사용하고 주석처리해놨습니다. 

movies = pd.read_csv("C:/Users/오승현/Desktop/2025-2/캡스톤디자인/movies.csv")
ratings = pd.read_csv("C:/Users/오승현/Desktop/2025-2/캡스톤디자인/ratings.csv")
users = pd.read_csv("C:/Users/오승현/Desktop/2025-2/캡스톤디자인/users.csv")

# print(movies.head())
# print(ratings.head())
# print(users.head())


import datetime

ratings = ratings.sort_values(['userId' , 'timestamp'])
ratings['datetime'] = ratings['timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x))
# print(ratings.head(10))



# 여기까지가 필요한거 import하고 data불러오고 그걸 시간순으로 정렬하는것까지 진행한거고 이 아래부턴
#  userId별로 어떤 영화를 어떤 순서로 평가했는지 묶고 
# 영화Id를 숫자 인덱스로 바꾸는 작업을 진행했습니다. 


user_sequences = ratings.groupby('userId')['movieId'].apply(list)

print(len(user_sequences))
print(user_sequences.head(10))

encoder = LabelEncoder()
ratings['movieId_encoded'] = encoder.fit_transform(ratings['movieId'])

user_sequences = ratings.groupby('userId')['movieId_encoded'].apply(list)
print(user_sequences.head(10))

# 아래는  모델이 배울 수 있는 형태로 데이터를 자르는 부분과 
# sequence 길이 맞추는 작업을 진행했습니다.


X = []
y = []

for seq in user_sequences:
    if len(seq) < 2:
        continue
    for i in range(1, len(seq)):
        X.append(seq[:i])
        y.append(seq[i])

        # for문을 이용했고 seq길이가 있을때 처음에 1개 2개 3개 4개 해서 i번쨰까지를 넣고 y에는 i를 넣어서 
        # x에는 입력값 지금까지 본 영화들을 넣고 y에는 다음 영화인 정답을 넣을 수 있게 진행했습니다. 

X = tf.keras.preprocessing.sequence.pad_sequences(X,padding='pre')
# 길이를 맞추기위해서 입력 시퀀스 길이를 자동으로 가장 긴 것에 맞춰서 패딩을 했습니다. 앞쪽을 다 0으로 채우게했습니다. 
X = np.array(X)
y = np.array(y)

print("전처리 완료")
print("X shape:", X.shape)
print("y shape:", y.shape)
print(X[:3])
print(y[:3])



# from sklearn.model_selection import train_test_split 도 위에 적어놨습니다.
# 이제 전처리 마지막 단계로 train/test/validation을 나누고  아직 gRU쓸지 LSTM쓸지 RNN쓸지 
# 안정했지만 LSTM쓴다고하면 LSTM모델 만들어서 model.fit으로 학습 진행하는단계만 만들면 될거같습니다. 
# train은 80% val 10% test 10%로 진행했습니다. 


# 먼저 train과 test를 나눔
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

# 남은 20% 중 절반을 validation으로 사용
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("훈련 데이터 크기:", X_train.shape)
print("검증 데이터 크기:", X_val.shape)
print("테스트 데이터 크기:", X_test.shape)

# Embedding + Encoder(GRU) + Bahdanau Attention + Decoder(GRU) 모델 구현 


num_movies = ratings['movieId_encoded'].nunique() + 1   # 영화 개수 + 시작토큰 1개 추가
START_TOKEN = num_movies - 1                             # 마지막 번호를 START_TOKEN으로 설정 # 전체 아이템 개수
sequence_length = X.shape[1]                       # 입력 시퀀스 길이
embedding_dim = 32
encoder_units = 64
decoder_units = 64
batch_size = 256


# ============================
# 2. Bahdanau Attention Layer
# ============================
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query : decoder hidden state (batch, hidden)
        # values: encoder outputs (batch, seq_len, hidden)

        query_with_time_axis = tf.expand_dims(query, 1)

        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(query_with_time_axis)
        ))

        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, tf.squeeze(attention_weights, -1)

# ============================
# 3. Encoder
# ============================
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(enc_units, return_sequences=True, return_state=True)

    def call(self, x):
        x = self.embedding(x)
        output, state = self.gru(x)
        return output, state
# ============================
# 4. Decoder
# ============================
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.attention = BahdanauAttention(dec_units)
        self.gru = tf.keras.layers.GRU(dec_units, return_sequences=True, return_state=True)
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, x, hidden, encoder_outputs):
        # 1 step decoder input
        x = self.embedding(x)                           # (batch, 1, embed)

        context_vector, attention_weights = self.attention(hidden, encoder_outputs)

        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        output, state = self.gru(x)

        output = self.fc(tf.reshape(output, (-1, output.shape[2])))

        return output, state, attention_weights

# ============================
# 5. 모델 구성
# ============================
encoder = Encoder(num_movies, embedding_dim, encoder_units)
decoder = Decoder(num_movies, embedding_dim, decoder_units)



train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(10000).batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_dataset = val_dataset.batch(batch_size)




optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

@tf.function
def train_step(enc_in, y):
    loss = 0

    with tf.GradientTape() as tape:
        # Encoder
        enc_output, enc_hidden = encoder(enc_in)

        dec_input = tf.fill((enc_in.shape[0], 1), START_TOKEN)


        # Decoder 1-step
        predictions, dec_hidden, attention_w = decoder(dec_input, enc_hidden, enc_output)

        loss = loss_object(y, predictions)

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return loss



EPOCHS = 5

for epoch in range(EPOCHS):
    total_loss = 0
    steps = 0

    for enc_in, target in train_dataset:
        batch_loss = train_step(enc_in, target)
        total_loss += batch_loss
        steps += 1

    print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss / steps}')

