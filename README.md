# Wiserli Notebooks

Colab notebooks for running Wiserli's YOLO model training and SAM 2 image-segmentation services on a temporary GPU runtime.

## Notebooks

| Notebook | What it does | Open in Colab |
| --- | --- | --- |
| [YOLOvX Training Pipeline](notebooks/YOLOvX_training_pipeline.ipynb) | Starts the FastAPI worker used by the Wiserli training page. Supports training a new model or fine-tuning an existing model. | [Open in Colab](https://colab.research.google.com/github/wiserli/notebooks/blob/main/notebooks/notebooks/YOLOvX_training_pipeline.ipynb) |
| [YOLOvX SAM Server](notebooks/YOLOvX_sam_server.ipynb) | Starts a SAM 2 server for point-prompted image segmentation and polygon results. | [Open in Colab](https://colab.research.google.com/github/wiserli/notebooks/blob/main/notebooks/notebooks/YOLOvX_sam_server.ipynb) |

The training worker source, [`finetune_worker.py`](finetune_worker.py), is downloaded by the training notebook when it runs. You normally do not need to start it separately.

## Run a Notebook

1. Open the notebook in Colab and choose **Runtime > Change runtime type > T4 GPU**.
2. Run the notebook cells from top to bottom. The setup cells install dependencies and download the required model files.
3. Leave the runtime running while your Wiserli app uses the service.
4. Copy the generated `trycloudflare.com` URL into the corresponding page in the app. The training notebook is for the Training page; the SAM notebook URL is used by the app's segmentation workflow.

Each session creates a temporary public URL. It changes when the tunnel or runtime restarts, and the service is unavailable after the runtime stops. Treat the URL as sensitive while the notebook is running; anyone who has it may be able to reach the exposed service.

