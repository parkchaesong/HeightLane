# HeightLane

Official implementation of **HeightLane** (WACV 2025 Oral Presentation)

---

### Overview

HeightLane is a novel approach for lane detection in Bird's-Eye-View (BEV) that integrates height information for improved performance. Our model extends the method introduced in [BEV-LaneDet](https://github.com/gigo-team/bev_lane_det).

---

### Installation

#### 1. Clone this repository:

```bash
git clone https://github.com/[YOUR_GITHUB]/HeightLane.git
```

#### 2. Clone the required dependency (Deformable-DETR) at the parent directory:

```bash
cd ..
git clone https://github.com/fundamentalvision/Deformable-DETR.git
```

#### 3. Compile CUDA operators:

Navigate to the operators directory and compile the necessary CUDA operators:

```bash
cd ./HeightLane/models/ops
sh ./make.sh
```

#### 4. Install Python dependencies:

Return to the HeightLane root directory and install the required packages:

```bash
cd ../../
pip install -r requirement.txt
```

---

### Usage

Detailed instructions for training and evaluation coming soon.

---

### Citation

If you use HeightLane in your research, please cite:

```
@inproceedings{YourPaper2025,
  title={HeightLane: Incorporating Height Information for Improved BEV Lane Detection},
  author={Your Name and Co-authors},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2025}
}
```

---

### Acknowledgments

This repository is built upon [BEV-LaneDet](https://github.com/gigo-team/bev_lane_det) and [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR). We sincerely thank the authors for their open-source contributions.

