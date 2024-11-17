
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
import shutil
import httpx
import uvicorn
import os
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_DIRECTORY = "uploads/"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
    print(file.filename)

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    async with httpx.AsyncClient() as client:
        await client.post("http://127.0.0.2:8001/send_filename", json={"filename":file.filename})
    return JSONResponse({"status": "success", "filename": file.filename})

@app.post("/start_stream")
async def start_stream(request: Request):
    data = await request.json()
    ip = data.get("ip")
    port = data.get("port")

    async with httpx.AsyncClient() as client:
        response = await client.post("http://127.0.0.2:8001/send_ip", json={"ip": ip, "port": port})
    return {"status": "success"} if response.status_code == 200 else {"status": "failed"}

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
