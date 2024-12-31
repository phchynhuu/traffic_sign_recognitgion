# traffic_sign_recognitgion
# GLARE_Dataset [[Paper](https://ieeexplore.ieee.org/document/10197287)] [[arXiv](https://arxiv.org/abs/2209.08716)]

Landing repository for GLARE Dataset and associated files from "GLARE: A Dataset for Traffic Sign Detection in Sun Glare". The dataset and associated files are released under [Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/legalcode).

[Link to the Dataset (Stored on Google Drive)](https://drive.google.com/drive/folders/1gmoOSgvjR4DP7jGfGS_xAmxcMShyeThx?usp=sharing).

Currently the dataset, tracks, and checkpoints files with logs have been uploaded. The folder Images contains the GLARE Dataset images and associated annotations, the folder Tracks contains video tracks, as discussed in the publication, and the folder Checkpoints contains the model checkpoint files and associated logs of the testing benchmarks. Not all the checkpoints are uploaded currently.

# Mapillary_Dataset
A diverse street-level imagery dataset with pixel‑accurate and instance‑specific human annotations for understanding street scenes around the world.

This Mapillary Vistas Dataset is provided under the Creative Commons Attribution NonCommercial Share Alike (CC BY-NC-SA) license.

[Link to the Dataset (Stored on Website)](https://www.mapillary.com/dataset/trafficsign)

# Guide to train

## Download dataset and push it to this folder

## Process format data for suitable with require of YOLO
- Process mapillary dataset

  ```bash
  python process_data_mapillary.py
  ```

- Process glare dataset

  ```bash
  python process_data_glare.py
  ```

## Train model

1. **Train YOLO Model**:
   - Use the default YOLO model for traffic sign recognition.

2. **Replace YOLO Head with ResNet**:
   - Optionally replace the YOLO head layer with a ResNet-18 architecture to modify the model's behavior.

## Arguments

The script accepts the following arguments:

| Argument         | Default Value          | Description                                                                 |
|------------------|------------------------|-----------------------------------------------------------------------------|
| `--model`        | `yolov9c.pt`          | Path to the YOLO model file (local or URL).                                |
| `--data`         | `./datasets/data.yaml`| Path to the `data.yaml` file.                                              |
| `--epochs`       | `1`                   | Number of training epochs.                                                 |
| `--imgsz`        | `640`                 | Image size for training.                                                   |
| `--batch`        | `16`                  | Batch size for training.                                                   |
| `--workers`      | `4`                   | Number of workers for data loading.                                        |
| `--device`       | `cuda`                | Device to use (`cuda` or `cpu`).                                           |
| `--output`       | `runs/train`          | Directory to save training results.                                        |
| `--cache`        | `False`               | Cache images for faster training.                                          |
| `--amp`          | `False`               | Use Automatic Mixed Precision (AMP) for training.                         |
| `--use_resnet`   | `False`               | Replace YOLO's head layer with ResNet.                                     |

## Usage

### Training with Default YOLO Head

To train the YOLO model with its default head layer, run:
```bash
python train_script.py --model yolov9c.pt --data ./datasets/data.yaml --epochs 10
```

### Training with ResNet Head

To replace the YOLO head layer with a ResNet-18 model, use the `--use_resnet` argument:
```bash
python train_script.py --model yolov9c.pt --data ./datasets/data.yaml --epochs 10 --use_resnet
```

### Additional Options

- Train on CPU:
  ```bash
  python train_script.py --device cpu
  ```

- Use cached images for faster training:
  ```bash
  python train_script.py --cache
  ```

- Enable mixed precision training for faster computation:
  ```bash
  python train_script.py --amp
  ```

## Output

- Training results, including model checkpoints and logs, are saved in the directory specified by `--output`. By default, this is `runs/train`.

## Notes

- Ensure your `data.yaml` file is correctly configured for the dataset.
- Adjust `epochs`, `imgsz`, and `batch` based on your system's resources and dataset size.
- The `--use_resnet` option modifies the YOLO architecture, which might affect performance.

*Created by Phuc Pham Huynh - [phucph.18@grad.uit.edu.vn](mailto:phucph.18@grad.uit.edu.vn)*
