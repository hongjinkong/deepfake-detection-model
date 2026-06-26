# 🔍 DeepFake Detection Model

> EfficientNet-B0 + Transformer 기반 딥페이크 탐지 모델 with Grad-CAM 시각화 웹앱

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12-red)
![Accuracy](https://img.shields.io/badge/Val%20Accuracy-97.64%25-brightgreen)

---

## 📌 프로젝트 개요

EfficientNet-B0을 Feature Extractor로, Transformer Encoder를 분류기로 사용하는 하이브리드 딥페이크 탐지 모델입니다.
단순한 모델 구현을 넘어 **Grad-CAM 히트맵 시각화 웹앱**까지 구현하여 AI가 이미지 어느 부분을 보고 판단했는지 시각적으로 확인할 수 있습니다.

---

## 🧠 모델 구조

```
입력 이미지 (224x224)
        ↓
EfficientNet-B0 (Feature Extractor)
        ↓
[B, 1280, 7, 7] → Flatten → [B, 49, 1280]
        ↓
Linear Projection → [B, 49, 512]
        ↓
Transformer Encoder Block x2 (Multi-Head Attention)
        ↓
Global Average Pooling → [B, 512]
        ↓
FC Layer → Binary Classification (Real / Fake)
```

---

## 📊 성능

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1 | 0.0929 | 96.53% | 0.1064 | 95.07% |
| 2 | 0.0578 | 97.86% | 0.0919 | 96.33% |
| 3 | - | 98%+ | - | **97.64%** |

---

## 📁 데이터셋

**140k Real and Fake Faces** (Kaggle)

- 출처: [Kaggle - 140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
- StyleGAN으로 생성된 가짜 얼굴 이미지

| Split | Real | Fake | 합계 |
|-------|------|------|------|
| Train | 70,001 | 70,001 | 140,002 |
| Validation | 19,787 | 19,641 | 39,428 |
| Test | 5,413 | 5,492 | 10,905 |

데이터셋 구조:
```
Dataset/
├── Train/
│   ├── Real/  (*.jpg)
│   └── Fake/  (*.jpg)
├── Validation/
│   ├── Real/
│   └── Fake/
└── Test/
    ├── Real/
    └── Fake/
```

---

## 🚀 실행 방법

### 1. 패키지 설치

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install fastapi uvicorn python-multipart pillow opencv-python
```

### 2. 데이터셋 준비

Kaggle에서 **140k Real and Fake Faces** 다운로드 후 아래 경로에 배치:

```
딥페이크탐지모델/
└── Dataset/
    ├── Train/
    ├── Validation/
    └── Test/
```

### 3. 모델 학습

`deepfake.ipynb` 실행 (순서대로 셀 실행)

```
Cell 1: 데이터 로더
Cell 2: 모델 정의
Cell 3: 학습 루프
Cell 4: Grad-CAM 테스트
```

학습 완료 후 `best_model.pth` 자동 저장됨

### 4. 웹앱 실행

```bash
python app.py
```

브라우저에서 `http://localhost:8000` 접속

---

## 🌐 웹앱 기능

| 기능 | 설명 |
|------|------|
| 이미지 업로드 | JPG, PNG 지원 |
| REAL/FAKE 판정 | 빨강/초록 색상으로 직관적 표시 |
| 신뢰도 | 프로그레스바 % 표시 |
| Grad-CAM 히트맵 | AI가 주목한 영역 시각화 |
| 오버레이 | 원본 이미지 위에 히트맵 합성 |

---

## ⚙️ 기술 스택

| 분류 | 기술 |
|------|------|
| 딥러닝 | PyTorch 2.12, EfficientNet-B0, Transformer |
| 학습 최적화 | AMP (Mixed Precision), AdamW, CosineAnnealingLR |
| 시각화 | Grad-CAM, OpenCV |
| 백엔드 | FastAPI, Uvicorn |
| 프론트엔드 | HTML, CSS, JavaScript |
| GPU | NVIDIA GeForce RTX 5060 Ti (CUDA 12.8) |

---

## 📂 프로젝트 구조

```
deepfake-detection-model/
├── deepfake.ipynb      # 학습 노트북
├── app.py              # FastAPI 백엔드
├── templates/
│   └── index.html      # 프론트엔드 UI
└── README.md
```

---

## ⚠️ 주의사항

본 모델은 **StyleGAN으로 생성된 딥페이크 이미지** 탐지에 최적화되어 있습니다.
Stable Diffusion, Midjourney 등 다른 생성 AI 이미지는 탐지 정확도가 낮을 수 있습니다.

---

## 🛠️ 개발 환경

- OS: Windows 11
- GPU: NVIDIA GeForce RTX 5060 Ti
- CUDA: 12.8
- Python: 3.11
- PyTorch: 2.12.0 (dev)