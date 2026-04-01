# =========================
# 1. import
# =========================
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import optim 


# =========================
# 2. 하이퍼 파라미터
# =========================
max_len = 50
hidden_dim = 50
num_heads = 2
num_layers = 2
batch_size = 128
LR = 0.001
epochs = 30

# =========================
# 3. 데이터 전처리
# =========================
ratings = pd.read_csv("ratings.csv")
ratings = ratings.sort_values(by=["userId", "timestamp"])

user_seq = ratings.groupby("userId")["movieId"].apply(list)

item_set = ratings["movieId"].unique()
item2idx = {item: i+1 for i, item in enumerate(item_set)}
idx2item = {i: item for item, i in item2idx.items()}

user_sequences = []
for seq in user_seq:
    user_sequences.append([item2idx[i] for i in seq])

# =========================
# 4. Dataset
# =========================
class SASRecDataset(Dataset):
    def __init__(self, sequences, max_len):
        self.sequences = sequences
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        seq = seq[-self.max_len:]
        input_seq = seq[:-1]
        target = seq[1:]

        pad_len = self.max_len - len(input_seq)
        input_seq = [0]*pad_len + input_seq
        target = [0]*pad_len + target

        return torch.LongTensor(input_seq), torch.LongTensor(target)

# =========================
# 5. 모델
# =========================
class SASRec(nn.Module):
    def __init__(self, num_items, hidden_dim=50, max_len=50, num_heads=2, num_layers=2):
        super(SASRec, self).__init__()

        self.item_emb = nn.Embedding(num_items+1, hidden_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_dim)

        self.layers = nn.ModuleList([ #nn.ModuleList는 여러 개의 layer를 "리스트처럼 저장"하는 PyTorch 전용 구조
            nn.TransformerEncoderLayer( # Input -> Multi-Head Attention(Q,K,V 생성) -> Feed Forward -> Output
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim*4,
                dropout=0.2,
                batch_first=True
            )
            for _ in range(num_layers)
        ])

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        batch_size, seq_len = x.size()

        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)

        x_emb = self.item_emb(x)
        x = x_emb + self.pos_emb(pos)
        x = self.dropout(x)

        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()

        padding_mask = (x_emb.sum(dim=-1) == 0)
        # padding_mask = (x == 0)
        for layer in self.layers:
            x = layer(x, src_mask=mask, src_key_padding_mask=padding_mask)

        x = self.layer_norm(x)

        return x

# =========================
# 6. 학습
# =========================
dataset = SASRecDataset(user_sequences, max_len)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SASRec(num_items=len(item2idx), max_len=max_len).to(device)

optimizer = optim.Adam(model.parameters(), lr=LR)


for epoch in range(epochs):
    model.train()
    total_loss = 0

    for input_seq, target in loader:
        input_seq = input_seq.to(device)
        target = target.to(device)

        output = model(input_seq)

        output = output.reshape(-1, output.size(-1))
        target = target.reshape(-1)

        # negative sampling
        neg_items = torch.randint(1, len(item2idx)+1, target.shape).to(device)

        pos_emb = model.item_emb(target)
        neg_emb = model.item_emb(neg_items)

        pos_logits = (output * pos_emb).sum(-1)
        neg_logits = (output * neg_emb).sum(-1)

        # padding 제외
        mask = target != 0

        pos_logits = pos_logits[mask]
        neg_logits = neg_logits[mask]

        loss = -torch.log(torch.sigmoid(pos_logits - neg_logits)).mean()

        optimizer.zero_grad() #기존 gradient 초기화 (0으로 reset)
        loss.backward() #미분계산 
        optimizer.step() # 실제로 파라미터 업데이트

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# =========================
# 7. 추천을 만들어내는 함수
# =========================
def recommend(model, user_sequence, top_k=10):
    model.eval()

    seq = user_sequence[-max_len:]
    seq = [0]*(max_len-len(seq)) + seq
    seq = torch.LongTensor(seq).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(seq)
        last = output[:, -1, :]

        scores = torch.matmul(last, model.item_emb.weight.T)
        top_items = torch.topk(scores, top_k).indices.squeeze().cpu().numpy()

    seen = set(user_sequence)
    return [idx2item[i] for i in top_items if i != 0 and idx2item[i] not in seen]
# for i in top_items 는 top_items중 i 즉 추천후보들중 0이아닌애들 0이면 가짜데이터니까 그리고  not in seen 은 이미 본 영화
#  이미 본 영화가 seen에 set으로 들어가있음 을 리턴해라top_items 안에 있는 i들 중에서
# → 0이 아니고 (padding 제외)
# → 이미 본 영화가 아닌 것만 골라서
# → 실제 영화 ID로 변환해서 반환한다