"""제공 모듈: 전처리와 기준선의 구성만 담당합니다. 이 파일은 수정하지 않습니다."""
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier


def build_pipeline():
    """아직 학습하지 않은 새 Pipeline을 돌려줍니다. fit은 실행 파일에서 호출합니다."""
    numeric = Pipeline([("fill", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([
        ("fill", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric, ["input_chars", "allowed_tools"]),
        ("cat", categorical, ["task_type", "priority"]),
    ])
    return Pipeline([("pre", preprocessor),
                     ("model", DummyClassifier(strategy="most_frequent"))])
