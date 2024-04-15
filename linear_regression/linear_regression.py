import torch
import torch.optim as optim

# 랜덤 시드를 줌으로써 실행을 여러번 해도 결과가 같도록 한다.
torch.manual_seed(1)

# 데이터
# 입력 값
x_train = torch.FloatTensor([[1], [2], [3]])
# 출력 값
y_train = torch.FloatTensor([[2], [4], [6]])

# 모델 초기화
# requires_grad란 학습을 통해 변경이 될 수 있는 값이라는 것을 알려주는 파라미터
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer 설정 ( SGD는 gradient descent의 종류 )
# lr은 학습률. 높으면 발산 / 낮으면 학습 속도가 느려짐
optimizer = optim.SGD([W, b], lr=0.01)

nb_epochs = 1999 # 원하는만큼 경사 하강법을 반복
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    hypothesis = x_train * W + b

    # cost 계산 (오차^2들의 합) => MSE. Mean Squared Error
    cost = torch.mean((hypothesis - y_train) ** 2)

    # 미분을 통해 얻은 기울기를 0으로 초기화. 초기화를 해야만 새로운 가중치 편향에 대해 새로운 기울기를 구할 수 있음
    optimizer.zero_grad()
    # 미분을 통해 기울기를 계산
    cost.backward()
    # W와 b에서 리턴되는 변수들의 기울기에 학습률(lr)을 곱하여 빼줌으로써 업데이트
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, W.item(), b.item(), cost.item()
        ))
        