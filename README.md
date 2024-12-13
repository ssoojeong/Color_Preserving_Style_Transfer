# &#x1F3A8; Color_Preserving_Style_Transfer

## &#x1F4E2; Project Report Overview: 241213
1. &#x2705; Style Transfer 비교 모델 조사
2. &#x1F680; 인퍼런스 실험 진행
    - &#x2705; (완료) Color Preserving Neural Style Transfer (CVPR 2016) 코드 정리
    - &#x2705; (완료) 실험 진행: Effeect 이미지 인퍼런스 테스트
    - &#x1F525; (예정) 환경 구축: Docker 생성
    - &#x1F525; (예정) 실험 진행: Effeect 이미지 인퍼런스 전수 완료

----

### &#x1F31F; 241213 Origial Dataset

```bash
# Original Effect(content) & Reference(style) 이미지
./dataset/effect/*.png 
./dataset/reference/*.png 
```

----

### &#x1F31F; Style Transfer 모델 리스트
- [Neural-Style-Transfer](https://github.com/rrmina/neural-style-pytorch.git) (CVPR 2016)

----

## &#x1F60E; 인퍼런스 실험 수행 Guide

### 1. Pretrained VGG 모델 다운로드
```bash
sh models/download_models.sh
```

### 2. Conda 환경 생성
```bash
conda create -n stpc python=3.8

conda activate stpc

# CUDA 11.3
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

pip install -r requirements.txt
```

### 3. Inference 실행
```bash
cd ./Effect_Generation_for_Children_Song

python inference.py
```

### 4. Output 확인
아래 폴더에 stylized image 생성

```bash
./outputs/reference_{style_name}/{content_name}.png 
```

----

## To do List
- &#x1F525; (예정) 환경 구축: Docker 생성
- &#x1F525; (예정) 실험 진행: Effeect 이미지 인퍼런스 전수 완료


