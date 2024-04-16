import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# softmax 함수의 결과에 로그를 씌우는 것 => log_softmax
# F.log_softmax() + F.nll_loss() = F.cross_entropy()
