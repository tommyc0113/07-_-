"""실습 05 응용: train 수치2열만 결측 대체·표준화하여 K=3으로 묶습니다."""
from pathlib import Path
import pandas as pd
import numpy as np
HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
manifest = pd.read_csv(HERE / "data/split_manifest.csv")
train_ids = manifest.loc[manifest["split"] == "train"].sort_values("position")["request_id"].tolist()
valid_ids = manifest.loc[manifest["split"] == "valid"].sort_values("position")["request_id"].tolist()
train, valid = raw.loc[train_ids].copy(), raw.loc[valid_ids].copy()
FEATURES = ["input_chars", "allowed_tools", "task_type", "priority"]
NUM, CAT = FEATURES[:2], FEATURES[2:]
X_train, X_valid = train[FEATURES].copy(), valid[FEATURES].copy()
y_train, y_valid = train["failed"], valid["failed"]
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
cluster = Pipeline([
    # 결측값을 학습 데이터의 중앙값으로 대체합니다.
    ("imputer", SimpleImputer(strategy="median")),

    # 수치형 특성을 표준화해 특성별 값의 범위 차이를 줄입니다.
    ("scaler", StandardScaler()),

    # 데이터를 3개 군집으로 나눕니다.
    # n_init=10은 초기 중심점을 바꿔 10회 실행합니다.
    # random_state=42는 결과를 재현할 수 있도록 설정합니다.
    ("kmeans", KMeans(
        n_clusters=3,
        n_init=10,
        random_state=42
    ))
])

# TODO A05: 수치2열만으로 cluster를 학습하세요. failed나 elapsed_ms를 넣지 않습니다.
# failed와 elapsed_ms는 군집 학습에 사용하지 않습니다.
X_cluster = X_train[NUM]

# 결측값 대체 → 표준화 → KMeans 학습을 실행합니다.
cluster.fit(X_cluster)
# TODO 끝

labels = cluster["kmeans"].labels_  # 학습 데이터 각 행의 군집 번호를 가져옵니다.

# 학습 데이터의 수치형 특성을 결측값 대체 및 표준화합니다.
# 실루엣 점수 계산에 사용할 표준화된 데이터를 얻습니다.
z = cluster[:-1].transform(X_train[NUM])


# 표준화된 군집 중심점을 원래 수치 단위로 되돌립니다.
centers = cluster["scaler"].inverse_transform(cluster["kmeans"].cluster_centers_)
profile = pd.DataFrame(centers, columns=NUM)
profile.insert(0, "cluster", range(3))
# 각 군집에 포함된 데이터 행의 개수를 계산합니다.
profile["n"] = np.bincount(labels, minlength=3)

profile.to_csv(OUT / "cluster_profile.csv", index=False, encoding="utf-8-sig")
pd.DataFrame({"request_id": train_ids, "cluster": labels}).to_csv(OUT / "cluster_assignments.csv", index=False, encoding="utf-8-sig")
print(profile.round(3).to_string(index=False))
print("silhouette(표준화 공간):", round(silhouette_score(z, labels), 3))
print("군집 번호에는 실패·성공·우열의 뜻이 없습니다.")
