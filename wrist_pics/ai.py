import os, torch, torchvision, cv2, numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from torchvision.models import ResNeXt50_32X4D_Weights, MobileNet_V2_Weights
import torch.nn.functional as F
import numpy as np

class GestureDataset(Dataset):
    def __init__(self, base_path, categories, transform=None):
        self.base_path = base_path
        self.categories = categories
        self.transform = transform
        self.images = []
        self.labels = []

        for label, category in enumerate(categories):
            folder_path = os.path.join(base_path, category)
            for filename in os.listdir(folder_path):
                if filename.endswith(".BMP"):
                    img_path = os.path.join(folder_path, filename)
                    self.images.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('L')  # 그레이스케일로 로드
        if self.transform:
            img = self.transform(img)
        return img, label
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook 등록
        self._register_hooks()

    def _save_gradient(self, grad):
        self.gradients = grad

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _register_hooks(self):
        self.target_layer.register_forward_hook(self._forward_hook)
        # register_backward_hook 대신 register_full_backward_hook 사용
        self.target_layer.register_full_backward_hook(lambda module, grad_in, grad_out: self._save_gradient(grad_out[0]))

    def generate(self, input_image, target_class=None):
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        one_hot_output = torch.zeros_like(output)
        one_hot_output[0][target_class] = 1
        output.backward(gradient=one_hot_output, retain_graph=True)

        # Grad-CAM 계산
        gradients = self.gradients[0].cpu().data.numpy()  # [C, H, W]
        activations = self.activations[0].cpu().data.numpy()  # [C, H, W]
        
        weights = np.mean(gradients, axis=(1, 2))  # 채널별 평균 그레이디언트
        cam = np.zeros(activations.shape[1:], dtype=np.float32)  # [H, W]

        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        # ReLU 적용 및 정규화
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))  # 원본 이미지 크기로 리사이즈
        cam = cam / cam.max()  # 0~1로 정규화
        return cam

# 히트맵 시각화 함수
def visualize_gradcam(original_image, cam, alpha=0.4):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    original = np.float32(original_image.permute(1, 2, 0).cpu().numpy())  # [C, H, W] -> [H, W, C]
    overlay = heatmap * alpha + original * (1 - alpha)
    overlay = overlay / overlay.max()  # 정규화
    return np.uint8(255 * overlay)

# Grad-CAM 적용 및 시각화 예제 (5장 처리)
def apply_gradcam(model, test_loader, device, target_layer, num_images=5):
    grad_cam = GradCAM(model, target_layer)
    
    # 이미지 카운터
    img_count = 0
    
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        # 배치 내에서 이미지 개수만큼 반복
        for i in range(images.size(0)):
            if img_count >= num_images:  # 5장 처리 후 종료
                return
            
            input_image = images[i].unsqueeze(0)  # 단일 이미지에 대해 처리
            true_label = labels[i].item()
            
            # Grad-CAM 생성
            cam = grad_cam.generate(input_image)
            
            # 원본 이미지와 히트맵 오버레이
            visualized = visualize_gradcam(images[i], cam)
            
            # 파일 이름에 인덱스와 실제 라벨 포함
            filename = f"gradcam_result_{img_count}_label_{true_label}.jpg"
            cv2.imwrite(filename, visualized)
            print(f"Grad-CAM 결과가 '{filename}'로 저장되었습니다.")
            
            img_count += 1
        
        if img_count >= num_images:
            break

class ResNext(nn.Module):
    def __init__(self, num_classes=2, input_channels=1):
        super(ResNext, self).__init__()
        self.resnext = models.resnext50_32x4d(weights = ResNeXt50_32X4D_Weights.DEFAULT)

        self.resnext.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        in_features = self.resnext.fc.in_features
        self.resnext.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnext(x)

class MobileNet(nn.Module):
    def __init__(self, num_classes=2, input_channels=1):
        super(MobileNet, self).__init__()
        self.mobilenet = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

        self.mobilenet.features[0][0] = nn.Conv2d(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )

        in_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.mobilenet(x)
    
##########################################################################################################################

base_path = "./test"

categories = ["open", "fist"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation((-5, 5)),  # 회전 범위 확대
    transforms.ColorJitter(brightness=0.2),  # 밝기 변화 (그레이스케일 기준)
    transforms.RandomAffine(degrees=5, translate=(0.1, 0.1)),  # 이동 및 회전
])

# 정확도 평가 함수
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

############################################################################################

dataset = GestureDataset(base_path, categories, transform=transform)

# 데이터 분할 (80% 학습, 20% 테스트)
indices = list(range(len(dataset)))
train_indices, test_indices = train_test_split(indices, test_size=0.2, stratify=dataset.labels, random_state=2)

train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

# DataLoader 생성 (배치 크기 16)
train_loader = DataLoader(dataset, batch_size=4, sampler=train_sampler)
test_loader = DataLoader(dataset, batch_size=4, sampler=test_sampler)

# 모델, 손실 함수, 옵티마이저 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    print(f"Device set to {device} because CUDA is not available.")
    raise RuntimeError("This program requires a CUDA-enabled GPU to run.")
    
#model = MobileNet().to(device)
model = ResNext().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(dataset.categories)

# 학습
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
    
    if (running_loss/len(train_loader) <= 0.05) and (epoch >= 10):
        break

# 정확도 계산
accuracy = evaluate_model(model, test_loader, device)
print(f"Test Accuracy: {accuracy:.2f}%")

original_data_path = "./eval"

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 이미지 크기 64x64
    transforms.ToTensor(),        # 텐서로 변환 (0~1 정규화)
])

# 증강되지 않은 데이터셋 로드
original_dataset = GestureDataset(original_data_path, categories, transform=transform)

# DataLoader 생성
original_test_loader = DataLoader(original_dataset, batch_size=4)

# 평가
accuracy = evaluate_model(model, original_test_loader, device)
print(f"Evaluation Accuracy: {accuracy:.2f}%")

# Grad-CAM 적용 (ResNext의 경우 마지막 컨볼루션 레이어 지정)
if isinstance(model, ResNext):
    target_layer = model.resnext.layer4[-1]  # ResNext의 마지막 레이어
elif isinstance(model, MobileNet):
    target_layer = model.mobilenet.features[-1]  # MobileNet의 마지막 피처 레이어

print("Applying Grad-CAM...")
apply_gradcam(model, test_loader, device, target_layer, num_images=5)

# 원본 데이터에 대한 평가
accuracy = evaluate_model(model, original_test_loader, device)
print(f"Test Accuracy (Original Data): {accuracy:.2f}%")

if accuracy >= 90:
    torch.save(model.state_dict(), f"{model.__class__.__name__}_{accuracy:.2f}.pth")