# HeightLane

Official implementation of **HeightLane** (WACV 2025 Oral Presentation)

---

### Overview

HeightLane is a novel approach for lane detection in Bird's-Eye-View (BEV) that integrates height information for improved performance. Our model extends the method introduced in [BEV-LaneDet](https://github.com/gigo-team/bev_lane_det).

---

### 0. Dataset Preparation

#### Step 1: Download OpenLane Dataset

Follow the instructions from the [OpenLane Dataset README](https://github.com/OpenDriveLab/OpenLane/blob/main/data/README.md) to download the full dataset.

After downloading, your directory structure should look like:

```
<root>/openlane/
├── images/
├── training/
└── validation/
```

#### Step 2: Download Height Map Data

Download the height map data from the following link:
[https://147.46.111.77:1402/sharing/jplpr7ROl](https://147.46.111.77:1402/sharing/jplpr7ROl)

Unzip the archive to get a folder named `Openlane_height`. Inside it, you will find folders such as:

```
Openlane_height/
├── heightmap_training/
└── heightmap_validation/
```

Move these two folders into the previously created `openlane/` folder so that the final structure is:

```
<root>/openlane/
├── images/
├── training/
├── validation/
├── heightmap_training/
└── heightmap_validation/
```

#### Step 3: Update Configuration

Set the path to your `<root>/openlane` directory in `heightlane_config.py`:

```python
ROOT_DIR = "/path/to/your/root/openlane"
```

---

### Directory Structure

To ensure everything works correctly, clone the repositories under a common parent directory like this:

```
<your_workspace>/
├── Deformable-DETR/
└── HeightLane/
```

---

### Installation

#### 1. Clone this repository:

```bash
git clone https://github.com/parkchaesong/HeightLane.git
```

#### 2. Clone the required dependency (Deformable-DETR) **in the same parent directory**:

```bash
git clone https://github.com/fundamentalvision/Deformable-DETR.git
```

#### 3. Compile CUDA operators:

Navigate to the operators directory and compile the necessary CUDA operators:

```bash
cd Deformable-DETR/models/ops
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
@InProceedings{Park_2025_WACV,
    author    = {Park, Chaesong and Seo, Eunbin and Lim, Jongwoo},
    title     = {HeightLane: BEV Heightmap Guided 3D Lane Detection},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {1692-1701}
}
```

---

### Acknowledgments

This repository is built upon \[BEV-LaneDet]\(h
