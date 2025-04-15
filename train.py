import os, cv2, time, math, random
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from scipy.optimize import linear_sum_assignment
import torch, gc

import warnings, time
warnings.filterwarnings('ignore') 

gc.collect(); torch.cuda.empty_cache()

# 랜덤 시드 고정
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # GPU 사용 시

# 재현성을 위한 추가 설정
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
"""-------------------------------------------------------------"""

from method.earlystop import EarlyStopping
from method.loss_key import CombinedLoss, GaussianLoss
from method.cocoprocessing_ import COCOKeypointDataset


"""
▶ Main Network code
"""
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x
        
        
class KeypointModel(nn.Module):
    def __init__(self, num_keypoints, d_model=1024, nhead=8, num_layers=6):
        super(KeypointModel, self).__init__()
        self.num_keypoints = num_keypoints
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers  # 추가: num_layers 저장

        # ResNet50 백본 (마지막 fully connected 층 제외)
        self.resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        
        # 2048 → 1024 변환
        self.feature_projection = nn.Conv2d(2048, d_model, kernel_size=1)

        # Position encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        # 각 레이어별 compressed representation 생성
        self.compressed_representor = nn.Linear(d_model, num_keypoints * 2)
        
        # weight 생성기
        self.gate_generator = nn.Linear(d_model, num_layers)

    def forward(self, x):
        # ResNet 특징 추출
        x = self.resnet(x)
        b, c, h, w = x.size()
        
        # 2048 → 1024 변환
        x = self.feature_projection(x)  # (batch_size, 256, h, w)
        # print(x.shape)
        
        # Transformer 입력 형태로 변환
        x = x.view(b, self.d_model, -1).permute(2, 0, 1)  # (sequence_length, batch_size, d_model)
        
        # Position encoding 추가
        x = self.pos_encoder(x)
        
        # Transformer Encoder 통과
        transformer_output = self.transformer_encoder(x)
        

        compressed_representations = []  # 모든 레이어의 compressed_rep을 저장할 리스트

        for i, layer in enumerate(self.transformer_encoder.layers):
            # 각 레이어에서 self-attention 수행
            layer_output, _ = layer.self_attn(x, x, x, need_weights=False)
            
            # Batch 평균 계산
            keypoint_rep = layer_output.mean(dim=0)  # (batch_size, d_model)
            # print('layer feature:', keypoint_rep.shape)

            # Compressed representation 생성
            compressed_rep = F.softplus(self.compressed_representor(keypoint_rep))  # (batch_size, compressed_rep_dim)
            # print('Keypoint rep:', compressed_rep.shape)
            
            # 결과 저장
            compressed_representations.append(compressed_rep)

            # 마지막 레이어에서 gated weights 계산
            if i == len(self.transformer_encoder.layers) - 1:
                gated_weights = F.softmax(self.gate_generator(keypoint_rep), dim=-1)  # (batch_size, 6)
                # print("Gated weights:", gated_weights.shape)

        # 리스트를 텐서로 변환: (layer_size, batch_size, compressed_rep_dim)
        compressed_representations = torch.stack(compressed_representations, dim=0)
        # print('Final compressed representations:', compressed_representations.shape)
        
        # Gated sum을 위한 가중치 적용
        # gated_weights의 차원 맞추기: (batch_size, 6) -> (batch_size, 6, 1)
        gated_weights = gated_weights.unsqueeze(2) # (batch_size, 6, 1)
        # print(gated_weights)
        # compressed_representations의 차원 맞추기: (6, batch_size, compressed_rep_dim) -> (batch_size, 6, compressed_rep_dim)
        compressed_representations = compressed_representations.permute(1, 0, 2)  # (batch_size, 6, compressed_rep_dim)

        # Gated sum을 위한 가중치 적용
        gated_sum_result = torch.sum(compressed_representations * gated_weights, dim=1)
        # print("Gated sum result:", gated_sum_result.shape)

        final_keypoints = torch.relu(gated_sum_result)
        # print(final_keypoints)

        return final_keypoints


"""
▶ 데이터 입력하기
"""

# COCO 데이터셋 불러오기
coco_train = COCO('../TEST/annotations/re_key/train.json')
coco_val = COCO('../TEST/annotations/re_key/valid.json')
coco_test = COCO('../TEST/annotations/re_key/test.json')

image_dir_train = '../TEST/images/train/images'
image_dir_val = '../TEST/images/valid/images'
image_dir_test = '../TEST/images/test/images'

# 변환 정의
transform = transforms.Compose([
    transforms.Resize(300),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 데이터셋 생성
train_dataset = COCOKeypointDataset(coco_train, image_dir_train, transform)
val_dataset = COCOKeypointDataset(coco_val, image_dir_val, transform)
test_dataset = COCOKeypointDataset(coco_test, image_dir_test, transform)

# 시드 고정
seed = 42
torch.manual_seed(seed)

# 데이터로더 생성
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)



"""
▶ 학습 전 세팅
"""


goal = f'./loss_1024/droneKey_gau_{gau_setting}'
print(goal)
os.makedirs(goal, exist_ok=True)

# 하이퍼파라미터 설정
num_keypoints = 4
learning_rate = 0.00001
num_epochs = 100

# 학습 루프 내부에 EarlyStopping 적용
early_stopping = EarlyStopping(patience=10, min_delta=0.0001)

# 모델 및 입력 준비
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = KeypointModel(num_keypoints).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # Adam 대신 SGD 사용, CUDA out of memory error 발생시...

criterion = GaussianLoss_exp()
# criterion = nn.MSELoss().to(device)

# 최종 학습 루프 수정
train_losses = []
valid_losses = []

train_accs = []
valid_accs = []


best_val_loss = 1.0
best_val_acc = 0.0


"""
▶ 학습 진행하기
"""

for epoch in range(num_epochs):
    print(f"[Epoch {epoch+1}/{num_epochs}]")
    criterion.update_sigma(epoch, num_epochs, min_sigma=False)  # base_sigma 업데이트

    start_time = time.time()  # 에폭 시작 시간 기록
    
    # Set model to training mode
    model.train()
    
    running_loss = 0.0
    running_acc = 0.0
    
    # Wrap train_loader with tqdm
    train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", total=len(train_loader), unit='batch')
    
    for i, (inputs, keypoints, rotations, translations) in enumerate(train_progress_bar):     
        inputs, keypoints = inputs.to(device), keypoints.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        
        # 손실 함수 계산
        loss = criterion(outputs, keypoints)
        loss.backward(retain_graph=True)
        optimizer.step()  # Update weights

        # print(loss)
        
        # 손실 및 정확도 누적
        running_loss += loss.item()

        # Update progress bar
        train_progress_bar.set_postfix({'Train Loss': f'{loss.item():.4f}'})

    # 에폭 손실 및 정확도 계산
    epoch_loss = running_loss / len(train_loader) 
    epoch_acc = running_acc / len(train_loader)
    
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)
    
    print(f"  --TRAIN-- \n  Train Total Loss: {epoch_loss:.8f}")

    # Set model to evaluation mode for validation
    model.eval()

    valid_loss = 0.0
    valid_acc = 0.0
    
    # Wrap valid_loader with tqdm
    val_progress_bar = tqdm(valid_loader, desc=f"Epoch {epoch+1} Validation", total=len(valid_loader), unit='batch')
    
    # Disable gradient computation during validation
    with torch.no_grad():
        for inputs, keypoints, rotations, translations in val_progress_bar:
            inputs, keypoints = inputs.to(device), keypoints.to(device)
            outputs = model(inputs)
            
            # 손실 함수 계산
            loss_v = criterion(outputs, keypoints)
            
            # 손실 및 정확도 누적
            valid_loss += loss_v.item()
            
            # Update progress bar
            val_progress_bar.set_postfix({ 'Val Loss': f'{loss_v.item():.4f}'})
    
    valid_loss /= len(valid_loader)
    valid_acc /= len(valid_loader)

    valid_losses.append(valid_loss)
    valid_accs.append(valid_acc)
    
    print(f"  --VALIDATION--  \n  Validation Total Loss: {valid_loss:.8f} ")
    
    # 최고 정확도를 기준으로 모델 저장
    if valid_loss < best_val_loss:
        best_val_loss = valid_loss
        torch.save(model.state_dict(), f'./loss_1024/droneKey_gau_{gau_setting}/best_model.pth')
        print(f"New best model saved with validation loss: {best_val_loss:.4f}")
        
    end_time = time.time()  # 에폭 종료 시간 기록
    epoch_duration = end_time - start_time  # 에폭에 걸린 시간 계산

    # 시간/분/초 형식으로 변환
    hours, rem = divmod(epoch_duration, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"  Epoch Duration: {int(hours)}h {int(minutes)}m {int(seconds)}s\n")  # 에폭 소요 시간 출력

    torch.save(model.state_dict(), f'./loss_1024/droneKey_gau_{gau_setting}/last_model.pth')

"""
▶ 결과 저장하기(Loss)
"""
plt.figure(figsize=(10, 5))  # ALL
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(valid_losses) + 1), valid_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss - all')
plt.legend()
plt.grid(True)
plt.savefig(f'./loss_1024/droneKey_gau_{gau_setting}/loss_all.png')
plt.close()
