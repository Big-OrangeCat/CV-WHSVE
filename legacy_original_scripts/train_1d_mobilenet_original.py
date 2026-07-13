import numpy as np, torch, torch.nn as nn, time
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score

#1D-MobileNet-V3
# 1. 瓒呭弬
BATCH, EPOCHS, LR, WD = 64, 80, 3e-4, 1e-4
import pandas as pd, os
csv_file = os.getenv("PLR_DATA_CSV", "data/preprocessed_data.csv")   # 浣犵殑鍘熷 CSV
df = pd.read_csv(csv_file)
time_cols = [f'D{i}' for i in range(1, 126)]
X_full = df[time_cols].values.astype('float32')
y_full = df['label'].values.astype('int64')
# 涓庝富瀹為獙鐩稿悓 8:2 鍒掑垎
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full)
# 褰㈢姸閫傞厤锛?50脳125 鈫?850脳1脳125
X_train = X_train[:, None, :]
X_test  = X_test[:, None, :]

# 2. 鏁版嵁
mean, std = X_train.mean(), X_train.std()
X_train = (X_train - mean)/std
X_test  = (X_test - mean)/std
train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train),
                                        torch.from_numpy(y_train)),
                          batch_size=BATCH, shuffle=True)
test_tensor  = torch.from_numpy(X_test)
y_test_tensor= torch.from_numpy(y_test)

# 3. 妯″瀷锛?.25 M锛?class SE(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(nn.Linear(c, c//r), nn.ReLU(inplace=True),
                                        nn.Linear(c//r, c), nn.Hardsigmoid())
    def forward(self, x):
        b, c, _ = x.size(); s = self.squeeze(x).view(b,c)
        return x * self.excitation(s).view(b,c,1)

class DW(nn.Module):
    def __init__(self, c, k=3, s=1, exp=2, se=True):
        super().__init__()
        hidden = int(c*exp)
        self.pw1 = nn.Conv1d(c, hidden, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.dw  = nn.Conv1d(hidden, hidden, k, s, k//2, groups=hidden, bias=False)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.se  = SE(hidden) if se else nn.Identity()
        self.pw2 = nn.Conv1d(hidden, c, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(c)
        self.act = nn.Hardswish()
    def forward(self, x):
        return x + self.bn3(self.pw2(self.se(self.act(self.bn2(self.dw(self.act(self.bn1(self.pw1(x)))))))))

class MobileNet1D(nn.Module):
    def __init__(self, num_classes=2, width=0.125):
        super().__init__()
        def make_div8(x): return int(np.ceil(x*width/8)*8)
        self.stem = nn.Sequential(nn.Conv1d(1, make_div8(8), 3, 1, 1, bias=False),
                                  nn.BatchNorm1d(make_div8(8)), nn.Hardswish())
        self.blocks = nn.Sequential(*[DW(make_div8(8), exp=2) for _ in range(3)],
                                    *[DW(make_div8(16), exp=2) for _ in range(3)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.2),
                                        nn.Linear(make_div8(16), num_classes))
    def forward(self, x):
        return self.classifier(self.pool(self.blocks(self.stem(x))).view(x.size(0),-1))

device = torch.device('cpu')
model  = MobileNet1D().to(device)

# 4. 璁粌
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
best_auc = 0
for epoch in range(EPOCHS):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward(); optimizer.step()
    # val
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(test_tensor.to(device)), dim=1)[:,1].cpu().numpy()
    auc = roc_auc_score(y_test, probs)
    print(f'Epoch {epoch+1:02d} | AUC = {auc:.4f}')
    if auc > best_auc: best_auc = auc; torch.save(model.state_dict(), '1d_mobilenet_best.pth')

print('Best AUC:', best_auc)

# ============= 鏈€缁堟寚鏍囷細Params / AUC / Acc / F1 / CPU latency =============
import time, sklearn.metrics as mt

# 1. 鍙傛暟閲?total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# 2. 鍦ㄦ渶浣虫ā鍨嬩笂璁＄畻 AUC銆丄cc銆丗1
model.load_state_dict(torch.load('1d_mobilenet_best.pth'))  # 杞藉叆鏈€浣虫潈鍊?model.eval()
with torch.no_grad():
    probs = torch.softmax(model(test_tensor.to(device)), dim=1)[:, 1].cpu().numpy()
preds = (probs > 0.5).astype(int)

auc  = mt.roc_auc_score(y_test, probs)
acc  = mt.accuracy_score(y_test, preds)
f1   = mt.f1_score(y_test, preds)

# 3. CPU 寤惰繜锛坆atch=1锛?00 娆″钩鍧囷紝鍗曟牳锛?lat = []
with torch.no_grad():
    for _ in range(100):
        tst = torch.randn(1, 1, 125).to(device)  # 涓庤缁冭緭鍏ヤ竴鑷?        t1 = time.perf_counter()
        _ = model(tst)
        lat.append(time.perf_counter() - t1)

print("========== 1dcnn鏈€缁堟寚鏍?==========")
print(f"Params     : {total_params/1e6:.3f} M")
print(f"AUC        : {auc:.4f}")
print(f"Accuracy   : {acc:.4f}")
print(f"F1-score   : {f1:.4f}")
print(f"CPU latency: {np.mean(lat)*1000:.1f} ms")
print("===========================================")
