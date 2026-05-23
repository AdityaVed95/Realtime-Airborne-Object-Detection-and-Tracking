from ultralytics import YOLO

# Point to last.pt from the crashed run — update path as needed
model = YOLO("runs/detect/runs/aod_big_run_2/weights/last.pt")

# resume=True restores optimizer/scheduler/epoch state from the checkpoint.
# Override epochs and patience here — args.yaml edits are ignored on resume.

model.train(
    resume=True,
    epochs=100,       # was 20; trains from epoch 21 → 40
    patience=20,     # was 5; more runway before early stopping kicks in
)
