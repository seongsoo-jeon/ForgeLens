import os
import random
import shutil

def split_val_dataset(base_path, val_ratio=0.1):
    # 경로 설정
    train_base = base_path
    val_base = base_path.replace("deepfake", "val") # Training/val 폴더 생성
    
    categories = ['0_real', '1_fake']
    
    for cat in categories:
        train_dir = os.path.join(train_base, cat)
        val_dir = os.path.join(val_base, cat)
        
        # val 폴더가 없으면 생성
        if not os.path.exists(val_dir):
            os.makedirs(val_dir)
            print(f"Created: {val_dir}")

        # 파일 목록 불러오기
        files = [f for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, f))]
        
        # 10% 계산 및 무작위 선택
        val_count = int(len(files) * val_ratio)
        val_files = random.sample(files, val_count)
        
        print(f"Moving {val_count} files from {cat} to val...")

        # 파일 이동
        for f in val_files:
            src = os.path.join(train_dir, f)
            dst = os.path.join(val_dir, f)
            shutil.move(src, dst)

    print("Successfully split the dataset!")

if __name__ == "__main__":
    # 데이터셋의 실제 경로를 입력하세요.
    dataset_path = "dataset_path/Training/deepfake"
    split_val_dataset(dataset_path)