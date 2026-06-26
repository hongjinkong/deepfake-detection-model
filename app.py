import torch
import torch.nn as nn
import numpy as np
import cv2
import base64
import io
from PIL import Image
from pathlib import Path
from torchvision import models
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import uvicorn

# ─── 모델 정의 (학습때랑 동일하게) ──────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads,
                                           dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class DeepfakeDetector(nn.Module):
    def __init__(self, num_transformer_blocks=2, dropout=0.3):
        super().__init__()
        backbone = models.efficientnet_b0(weights=None)
        self.feature_extractor = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.patch_proj = nn.Linear(1280, 512)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(dim=512, num_heads=8, dropout=dropout)
              for _ in range(num_transformer_blocks)]
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x)
        B, C, H, W = x.shape
        x = x.flatten(2)
        x = x.permute(0, 2, 1)
        x = self.patch_proj(x)
        x = self.transformer_blocks(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x.squeeze(1)


# ─── Grad-CAM ────────────────────────────────────────────
class GradCAM:
    def __init__(self, model):
        self.model      = model
        self.gradient   = None
        self.activation = None
        target_layer = model.feature_extractor[-1]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activation = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradient = grad_output[0].detach()

    def generate(self, input_tensor):
        output = self.model(input_tensor)
        self.model.zero_grad()
        output.backward()
        weights = self.gradient.mean(dim=[2, 3], keepdim=True)
        cam     = (weights * self.activation).sum(dim=1, keepdim=True)
        cam     = torch.relu(cam)
        cam     = cam.squeeze().cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# ─── FastAPI 앱 초기화 ────────────────────────────────────
app       = FastAPI()
templates = Jinja2Templates(directory="templates")

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path(r"C:\Users\smhrd\Desktop\딥페이크탐지모델\best_model.pth")

# 모델 로드
model = DeepfakeDetector().to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"✅ 모델 로드 완료! (Device: {DEVICE})")

gradcam = GradCAM(model)

# 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ─── 메인 페이지 ──────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html"
    )


# ─── 이미지 분석 API ──────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 이미지 읽기
    img_bytes = await file.read()
    img_pil   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_pil   = img_pil.resize((224, 224))
    img_orig  = np.array(img_pil)

    # 추론
    input_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
    input_tensor.requires_grad_(True)

    cam  = gradcam.generate(input_tensor)
    prob = torch.sigmoid(model(transform(img_pil).unsqueeze(0).to(DEVICE))).item()

    is_fake = prob < 0.5
    label   = "FAKE" if is_fake else "REAL"
    conf    = round((1 - prob) * 100 if is_fake else prob * 100, 2)

    # 히트맵 오버레이 생성
    cam_resized = cv2.resize(cam, (224, 224))
    heatmap     = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap     = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay     = cv2.addWeighted(img_orig, 0.5, heatmap, 0.5, 0)

    # base64로 인코딩 (프론트엔드에 전송)
    def to_base64(arr):
        img = Image.fromarray(arr.astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    heatmap_color = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    return JSONResponse({
        "label"     : label,
        "confidence": conf,
        "original"  : to_base64(img_orig),
        "heatmap"   : to_base64(heatmap_color),
        "overlay"   : to_base64(overlay),
    })
      


# ─── 서버 실행 ────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)