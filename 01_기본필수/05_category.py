"""05 기본필수: qa와 summary를 서로 다른 0/1 열로 표현합니다."""
from sklearn.preprocessing import OneHotEncoder

train = [["qa"], ["summary"], ["qa"]]
valid = [["summary"], ["qa"]]
# sparse_output=False는 결과를 바로 볼 수 있는 숫자 표로 받는 설정입니다.
encoder = OneHotEncoder(sparse_output=False)
encoder.fit(train)

# TODO B05-1: encoder.transform(valid).tolist()로 실제 변환 결과를 구하세요.
valid_ready = encoder.transform(valid).tolist() 
print("열 순서:", encoder.categories_[0].tolist())
print("valid 변환:", valid_ready)
