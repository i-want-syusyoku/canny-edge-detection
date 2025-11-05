from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2
from PIL import Image
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 必要に応じて特定のドメインに制限
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    # ファイルのバイナリデータを読み込む
    image_data = await file.read()

    # バイナリデータを NumPy 配列に変換
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # グレースケール変換 & エッジ検出
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 150, 200)

    # エッジ画像をJPEG形式にエンコード
    _, buffer = cv2.imencode(".jpg", edges)

    # エンコードされた画像をBase64文字列に変換
    encoded_image = base64.b64encode(buffer).decode("utf-8")

    return JSONResponse(content={"image": encoded_image})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")