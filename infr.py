import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from models.network.net_stage2 import net_stage2
from options.options import Options

def run_inference():
    # 1. 설정 및 모델 로드
    options = Options()
    opt = options.parse()
    
    # 모델 정의 (Stage 2 권장)
    model = net_stage2(opt, train=False)
    
    # 학습된 가중치 로드 (가장 성능이 좋았던 Stage 2 체크포인트 경로)
    checkpoint_path = './check_points/train_setting_df/train_stage_2/model/model_best_val_loss.pth'
    model_load = torch.load(checkpoint_path)
    model.load_state_dict(model_load['model_state_dict'])
    
    model.cuda()
    model.eval()
    print(f"Loaded weights from {checkpoint_path}")

    # 2. 이미지 전처리 설정 (학습 시와 동일해야 함)
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # 모델 입력 사이즈 확인 필요
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. 이미지 리스트 생성 및 알파벳 순 정렬
    image_dir = './dataset_path/Training/eval/deepfake'
    img_names = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    img_names.sort() # 알파벳 순 정렬 (0.png, 1.png, 10.png, 11.png ... 순서)
    
    print(f"Total images found: {len(img_names)}")

    # 4. 추론 및 결과 저장
    probabilities = []

    with torch.no_grad():
        for name in tqdm(img_names):
            img_path = os.path.join(image_dir, name)
            img = Image.open(img_path).convert('RGB')
            img_t = transform(img).unsqueeze(0).cuda()

            # 모델 예측
            output = model(img_t)
            # Sigmoid를 적용하여 0.0~1.0 사이 확률값 추출
            prob = torch.sigmoid(output).item()
            probabilities.append(prob)

    # 5. submission.txt 저장
    with open('submission.txt', 'w') as f:
        for p in probabilities:
            f.write(f"{p:.6f}\n")
    
    print("Inference completed! 'submission.txt' has been created.")

if __name__ == '__main__':
    run_inference()