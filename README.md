# DroneKey
## DroneKey: Drone 3D Pose Estimation in Image Sequences using Gated Key-representation and Pose-adaptive Learning

![DroneKey Overview](./assets/overview_.gif)

**DroneKey** is a framework for estimating the 3D pose (6DoF) of drones from monocular image sequences. It detects the four propellers of the drone as keypoints and geometrically reconstructs the drone's position and orientation using a PnP solver. The framework consists of a Transformer-based keypoint detector and a geometry-based pose estimator, achieving both real-time performance and high accuracy.

## 🔍 Highlights

- **Drone-specific Keypoint Detection**
  - Extracts intermediate and compact representations from all Transformer encoder layers and integrates them using a gated sum mechanism to learn layer-wise importance.
  
- **Pose-Adaptive Mahalanobis Loss**
  - Adapts dynamically to pose variation and improves stability during training.
  
- **Real-Time and High Accuracy**
  - Achieves 44 FPS and 99.68% AP (OKS) using a lightweight 6-layer encoder.
  
- **Public Dataset Release**
  - Provides [DroneKey Dataset](https://sites.google.com/view/cvl-jnu/publication/dronekey/dronekey_dataset) including 2D keypoints and 3D poses captured with realistic 360° backgrounds.

## 📁 Dataset

- **2DroneKey**: 2D drone images with keypoint annotations (10K images, two drone types)
- **3DronePose**: Includes full 6DoF pose and keypoints for 3 additional sequences
- Download link:  
  👉 [https://sites.google.com/view/cvl-jnu/publication/dronekey/dronekey_dataset](https://sites.google.com/view/cvl-jnu/publication/dronekey/dronekey_dataset)

## 🧠 Architecture Overview
![DroneKey Architecture](./assets/architecture.gif)
1. **CNN + Transformer Encoder**: Extracts visual tokens and learns global relationships
2. **Keypoint Head**: Aggregates multi-level features via gated sum to predict keypoints
3. **3D Pose Estimator**: Computes 6DoF pose using PnP and drone specifications

## ⚙️ How to Run

```bash
# Set up environment
conda create -n dronekey python=3.9
conda activate dronekey
pip install -r requirements.txt

# Train
python train.py --config configs/dronekey.yaml

# Test
python test.py --model weights/best_model.pth
```

## 🧪 Performance Summary
![DroneKey Results](./assets/result_.gif)


## 📜 Citation
```bash
@misc{DroneKey2025,
  author = {Seo-Bin Hwang and Yeong-Jun Cho},
  title = {DroneKey: Drone 3D Pose Estimation in Image Sequences using Gated Key-representation and Pose-adaptive Learning},
  year = {2025},
  note = {Under review at IROS 2025},
  url = {https://github.com/kkanuseobin/DroneKey}
}
```

## 📩 Contact
Seo-Bin Hwang: cnu.cvl.hsb@gmail.com
Department of AI Convergence, Chonnam National University, Korea
