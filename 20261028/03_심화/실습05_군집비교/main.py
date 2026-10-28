"""실습 05 심화: 같은 표준화 공간에서 K=2·3·4를 비교하고 새 요청에 같은 규칙을 씁니다."""
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
rows = []
for k in [2, 3, 4]:
    # 결측값 처리, 표준화, KMeans 군집화를 순서대로 실행합니다.
    model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
                      ("cluster", KMeans(n_clusters=k, n_init=10, random_state=42))])
    # TODO H05: 현재 K의 model을 train 수치2열로 학습하세요. 비교할 때 전처리와 입력은 고정합니다.
    # X_train[NUM]은 input_chars와 allowed_tools만 선택합니다.
    # fit()을 호출하면 결측값 처리, 표준화, 군집화가 차례대로 학습됩니다.
    model.fit(X_train[NUM])
    # TODO 끝

    # 학습 데이터에 적용된 전처리 결과를 가져옵니다.
    # 실루엣 점수도 이 표준화된 공간에서 계산합니다.
    z = model[:-1].transform(X_train[NUM])

    # 각 학습 데이터가 속한 군집 번호를 가져옵니다.
    labels = model["cluster"].labels_
    
    rows.append({"K": k, "inertia": model["cluster"].inertia_, "silhouette": silhouette_score(z, labels)})
    if k == 3:
        selected = model
result = pd.DataFrame(rows)
result.to_csv(OUT / "cluster_scores.csv", index=False, encoding="utf-8-sig")
centers = selected["scaler"].inverse_transform(selected["cluster"].cluster_centers_)
profile = pd.DataFrame(centers, columns=NUM)
profile.insert(0, "cluster", range(3))
profile["n"] = np.bincount(selected["cluster"].labels_, minlength=3)
profile.to_csv(OUT / "cluster_profile.csv", index=False, encoding="utf-8-sig")
new = pd.DataFrame({"input_chars": [500, 2000], "allowed_tools": [1, 4]})
print(result.round(4).to_string(index=False))
print(profile.round(2).to_string(index=False))
print("새 요청의 K=3 군집:", selected.predict(new).tolist())
print("inertia는 표준화 공간의 제곱거리 합입니다. F1과 평균내지 않습니다.")
