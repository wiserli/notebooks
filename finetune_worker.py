import os
import sys
import json
import shutil
import zipfile
import requests
from typing import Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO

LOG_FILE_PATH = "/content/worker.log"

class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()
    def isatty(self):
        return self.streams[0].isatty() if hasattr(self.streams[0], "isatty") else False
    def fileno(self):
        return self.streams[0].fileno()
    @property
    def encoding(self):
        return getattr(self.streams[0], "encoding", "utf-8")

_log_file = open(LOG_FILE_PATH, "a", buffering=1)
sys.stdout = _Tee(sys.stdout, _log_file)
sys.stderr = _Tee(sys.stderr, _log_file)

app = FastAPI(title="YOLOvX Training Worker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = "https://base.wiserly.org"
LAST_RUN_PATH = "/content/last_run.json"

ALLOWED_BASE_ARCHITECTURES = {
    "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
    "yolov9t", "yolov9s", "yolov9m", "yolov9c", "yolov9e",
    "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x",
}


class FineTuneRequest(BaseModel):
    session_token: str
    model_id: str
    project_id: str
    version_tag: Optional[str] = None
    epochs: int = 30
    mode: str = "finetune"
    base_architecture: Optional[str] = None
    imgsz: int = 640
    batch: int = 16
    optimizer: str = "auto"
    lr0: Optional[float] = None
    lrf: Optional[float] = None
    momentum: Optional[float] = None
    weight_decay: Optional[float] = None
    warmup_epochs: Optional[float] = None
    warmup_momentum: Optional[float] = None
    warmup_bias_lr: Optional[float] = None
    box: Optional[float] = None
    cls: Optional[float] = None
    dfl: Optional[float] = None
    hsv_h: Optional[float] = None
    hsv_s: Optional[float] = None
    hsv_v: Optional[float] = None
    degrees: Optional[float] = None
    translate: Optional[float] = None
    scale: Optional[float] = None
    shear: Optional[float] = None
    perspective: Optional[float] = None
    flipud: Optional[float] = None
    fliplr: Optional[float] = None
    mosaic: Optional[float] = None
    mixup: Optional[float] = None
    copy_paste: Optional[float] = None
    patience: Optional[int] = None


class UploadRequest(BaseModel):
    session_token: str


@app.get("/health")
def health():
    return {"status": "ok", "worker": "YOLOvX Training Worker"}


def update_status(session_token: str, status: str, error_message: str = None, **extra):
    try:
        payload = {"session_token": session_token, "status": status, **extra}
        if error_message:
            payload["error_message"] = error_message[:500]
        requests.post(f"{SUPABASE_URL}/functions/v1/finetune-update-status", json=payload, timeout=10)
    except Exception as e:
        print(f"Failed to update status: {e}")


def fetch_user_base_model(session_token: str, model_name: str) -> str:
    res = requests.post(
        f"{SUPABASE_URL}/functions/v1/finetune-fetch-model",
        json={"session_token": session_token, "model_name": model_name},
    )
    if res.status_code != 200:
        try:
            detail = res.json().get("error", res.text)
        except Exception:
            detail = res.text
        raise Exception(f"Failed to fetch user base model: {detail}")

    local_path = f"{model_name}_base.pt"
    with open(local_path, "wb") as f:
        f.write(res.content)
    if os.path.getsize(local_path) == 0:
        raise Exception("Decrypted base model was empty")
    return local_path


def resolve_base_weights(req: "FineTuneRequest", model_id: str) -> str:
    if req.mode == "new":
        arch = (req.base_architecture or "").strip().lower()
        if arch not in ALLOWED_BASE_ARCHITECTURES:
            raise Exception(
                f"Invalid or missing base_architecture '{arch}' for mode='new'. "
                f"Must be one of: {sorted(ALLOWED_BASE_ARCHITECTURES)}"
            )
        return f"{arch}.pt"
    else:
        path = fetch_user_base_model(req.session_token, model_id)
        return path


def fetch_dataset(session_token: str, project_id: str, version_tag: Optional[str] = None) -> str:
    payload = {"session_token": session_token, "project_id": project_id}
    if version_tag:
        payload["version_tag"] = version_tag
    res = requests.post(
        f"{SUPABASE_URL}/functions/v1/finetune-export-dataset",
        json=payload,
    )
    if res.status_code != 200:
        try:
            detail = res.json().get("error", res.text)
        except Exception:
            detail = res.text
        raise Exception(f"Failed to export dataset: {detail}")

    dataset_dir = "/content/dataset"
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    os.makedirs(dataset_dir, exist_ok=True)

    zip_path = "/content/dataset.zip"
    with open(zip_path, "wb") as f:
        f.write(res.content)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dataset_dir)
    os.remove(zip_path)

    data_yaml_path = os.path.join(dataset_dir, "data.yaml")
    if not os.path.exists(data_yaml_path):
        raise Exception("data.yaml missing from exported dataset")
    return data_yaml_path


def run_finetune_job(req: "FineTuneRequest", model_id: str):
    base_weights_path = None
    is_downloaded_base = False
    update_status(
        req.session_token, "training",
        project_id=req.project_id, model_id=model_id, mode=req.mode,
        base_architecture=req.base_architecture if req.mode == "new" else None,
        parent_model=model_id if req.mode == "finetune" else None,
        epochs=req.epochs, dataset_version=req.version_tag or "v0",
    )
    try:
        base_weights_path = resolve_base_weights(req, model_id)
        is_downloaded_base = req.mode != "new"

        data_yaml_path = fetch_dataset(req.session_token, req.project_id, req.version_tag)

        train_kwargs = {
            "data": data_yaml_path,
            "epochs": req.epochs,
            "imgsz": req.imgsz,
            "batch": req.batch,
            "optimizer": req.optimizer,
        }
        optional_fields = [
            "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs",
            "warmup_momentum", "warmup_bias_lr", "box", "cls", "dfl",
            "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale",
            "shear", "perspective", "flipud", "fliplr", "mosaic", "mixup",
            "copy_paste", "patience",
        ]
        for field in optional_fields:
            val = getattr(req, field)
            if val is not None:
                train_kwargs[field] = val

        model = YOLO(base_weights_path)
        results = model.train(**train_kwargs)

        weights_path = os.path.join(results.save_dir, "weights", "best.pt")
        if not os.path.exists(weights_path):
            raise Exception("Weights file not found after training completed.")

        metrics = {}
        try:
            rd = results.results_dict
            metrics = {
                "mAP50": rd.get("metrics/mAP50(B)"),
                "mAP50-95": rd.get("metrics/mAP50-95(B)"),
                "precision": rd.get("metrics/precision(B)"),
                "recall": rd.get("metrics/recall(B)"),
            }
        except Exception:
            pass

        with open(LAST_RUN_PATH, "w") as f:
            json.dump({
                "session_token": req.session_token,
                "model_id": model_id,
                "mode": req.mode,
                "weights_path": weights_path,
                "metrics": metrics,
                "epochs": req.epochs,
                "train_kwargs": {k: v for k, v in train_kwargs.items() if k != "data"},
            }, f)

        update_status(
            req.session_token, "review",
            hyperparameters={k: v for k, v in train_kwargs.items() if k != "data"},
            metrics=metrics,
        )
    except Exception as e:
        update_status(req.session_token, "failed", str(e))
    finally:
        if is_downloaded_base and base_weights_path and os.path.exists(base_weights_path):
            os.remove(base_weights_path)


@app.post("/start-finetune")
def start_finetune(req: FineTuneRequest, background_tasks: BackgroundTasks):
    verify_res = requests.post(
        f"{SUPABASE_URL}/functions/v1/verify-finetune-session",
        json={"session_token": req.session_token}
    )
    if verify_res.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid or expired session token")
    session_data = verify_res.json()

    if req.mode == "new":
        model_id = req.model_id
    else:
        model_id = session_data.get("model_id", req.model_id)

    background_tasks.add_task(run_finetune_job, req, model_id)

    return {
        "status": "started",
        "message": "Training dispatched.",
        "model_id": model_id,
        "mode": req.mode,
    }


@app.post("/upload-weights")
def upload_weights(req: UploadRequest):
    if not req.session_token:
        raise HTTPException(status_code=400, detail="session_token is required")

    if not os.path.exists(LAST_RUN_PATH):
        raise HTTPException(status_code=404, detail="No completed training run available.")

    try:
        with open(LAST_RUN_PATH, "r") as f:
            run_data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read metadata: {e}")

    staged_token = run_data.get("session_token")
    if not staged_token or staged_token != req.session_token:
        raise HTTPException(status_code=403, detail="Session does not match staged weights.")

    weights_path = run_data.get("weights_path")
    if not weights_path or not os.path.exists(weights_path):
        raise HTTPException(status_code=404, detail="Training weights file not found.")

    model_id = run_data.get("model_id")
    if not model_id:
        raise HTTPException(status_code=500, detail="Training run missing model_id.")

    try:
        with open(weights_path, "rb") as weights_file:
            response = requests.post(
                f"{SUPABASE_URL}/functions/v1/complete-finetune-session",
                headers={"Authorization": f"Bearer {req.session_token}"},
                files={"file": ("best.pt", weights_file, "application/octet-stream")},
                data={"model_id": model_id},
                timeout=120,
            )
    except requests.RequestException:
        raise HTTPException(status_code=502, detail="Could not connect to upload service.")

    if response.status_code != 200:
        raise HTTPException(status_code=502, detail="Weight upload failed.")

    try:
        upload_result = response.json()
    except Exception:
        upload_result = {"success": True}

    try:
        update_status(req.session_token, "completed")
    except Exception:
        pass

    try:
        os.remove(LAST_RUN_PATH)
    except Exception:
        pass

    return {
        "success": True,
        "model_id": model_id,
        "message": "Weights uploaded successfully.",
        "result": upload_result,
    }
