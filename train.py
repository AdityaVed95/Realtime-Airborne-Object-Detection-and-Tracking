import os

# Ultralytics has historically silently dropped workers to 0 on MPS due to
# fork/multiprocessing quirks. Enabling the MPS fallback env var before the
# ultralytics import makes parallel decode more reliable.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from ultralytics import YOLO

# Pretrained COCO weights — same starting point as train.py
model = YOLO("yolo26s.pt")

results = model.train(
    # ---- what & where ----
    data="data.yaml",
    project="runs",
    name="aod_big_run_2",

    # ---- compute ----
    device="mps",
    amp=False,           

    # ---- throughput levers (the speedup) ----
    batch=128,           
    imgsz=640,           # same 640 as before; matches pretrained weights
    workers=4,          
    cache="ram",         # preload decoded dataset (~19 GB) into RAM; paid once, not per epoch
    
    # ---- schedule ----
    epochs=100,
    patience=20,          # stop if no mAP improvement for 5 epochs
    save_period=1,       # save checkpoint every epoch so a crash doesn't lose progress
    close_mosaic=10,     # default; disables mosaic for final 10 epochs for clean convergence

)
