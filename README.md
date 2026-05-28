# Realtime-Airborne-Object-Detection-and-Tracking

### Description of each file : 

**`a1989895_Final_Project_Report_DeepLearning_Applications_Real-Time-Airborne-Object-Detection-and-Tracking.pdf`** - Final Project Report Doc

**`AdityaVed_a1989895_Deep_Learning_Assignment2_Airborne_Object_Detection.pdf`** - Presententation Deck for Assignment 2

**`a1989895_Deep_Learning_Applications_Assignment_3.pdf`** - Presententation Deck for Assignment 3

**`video_explanation.txt`** - Youtube Video Link for explanation

**`train.py`** — Trains a YOLO26s model on the airborne object dataset. Loads pretrained COCO weights, sets MPS (Apple Silicon) as the device, caches the dataset in RAM, and runs 100 epochs at imgsz 640 with batch size 128. Saves checkpoints every epoch to `runs/aod_big_run_2/`.

**`train_resume.py`** — Resumes a previously interrupted training run from `last.pt`, restoring optimizer and epoch state, and extends training to 100 epochs with a longer early-stopping patience of 20.

**`verify.py`** — Validates the YOLO dataset structure end to end. For each split (train/valid/test), it checks image-label pairing, opens images to confirm they aren't corrupt, validates that every label line is `class x y w h` with normalized coordinates in range, and prints class distribution plus empty-label counts.

**`verify_yolo_parse_dataset.py`** — One-liner sanity check that runs Ultralytics' built-in `check_det_dataset('data.yaml')` to confirm the dataset config parses correctly.

**`validate_all_splits_dimensions.py`** — Walks every JPG in train/valid/test and verifies the image dimensions are exactly 1024x1024, listing any mismatches.

**`dataset_analysis.ipynb`** — Exploratory data analysis of the dataset. Parses all label files once, then produces tables and plots for: annotations per class per split, images per class per split, bounding box width/height/area distributions, bbox center heatmaps per class, aspect ratio distributions, objects-per-image histogram, class co-occurrence matrix, and overall train/valid/test split proportions.

**`visualize_training.ipynb`** — Reads `100_epochs_results.csv` (the training log) and plots train vs val curves for box loss, classification loss, and DFL loss, plus mAP50, mAP50-95, and precision vs recall over epochs. Each plot is followed by markdown insights about overfitting, instability bumps, and final scores.

**`predict.ipynb`** — Runs inference with the fine-tuned `best.pt` on the test split. Predicts on a single random image, then on a 3x3 grid of random samples, then on the entire test split (saving annotated images to `test_split_prediction/`), and finally calls `model.val(split="test")` to compute mAP50, mAP50-95, precision, recall, and a per-class breakdown into `test_split_metrics/`.

**`video_object_tracking.ipynb`** — Realtime tracking on input videos using BoTSORT on top of the fine-tuned YOLO model. For each frame it draws bounding boxes with track IDs, a per-class count panel, a relative-depth indicator (approaching/receding/stable based on bbox area change rate via log-linear regression), and a rough metric distance estimate using the pinhole camera model with per-class assumed real-world sizes and a configured focal length. Writes the annotated video to `videos/output/`.