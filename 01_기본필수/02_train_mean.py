"""02 기본필수: 평균을 어디에서 계산해야 하는지 비교합니다."""
train_values = [10, 20]
valid_values = [100]

# 진짜 학습 기준으로 사용할 수 있는 것은 train의 평균입니다.
# TODO B02-1: train_values의 합을 개수로 나누세요.

train_mean = sum(train_values) / len(train_values)
# 아래 전체 평균은 '잘못된 기준과의 비교'를 위한 시연일 뿐입니다.
all_values = train_values + valid_values
all_mean = sum(all_values) / len(all_values)

print("train 평균:", round(train_mean, 2))
print("전체 평균(비교만):", round(all_mean, 2))
print("실제 전처리에는 train 평균만 사용합니다.")
