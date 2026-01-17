# Data Sources and Attribution

## Overview

This document describes all data sources used in the Ornimetrics Bird Detection System and confirms compliance with licensing and usage rights.

## ✅ Data Usage Summary

**All data in this project is either:**
1. **Synthetically generated** (procedural, no copyright)
2. **Public domain** (freely available scientific data)
3. **User-generated** (captured by your own cameras)

**✅ Safe for research purposes**
**✅ Safe for commercial use**
**✅ No copyright infringement**
**✅ No licensing restrictions**

---

## Point Cloud (PLY) Models

### Source: **100% Synthetic - Procedurally Generated**

**Location:** `data/species_prototypes/*.ply`

**Generation Method:**
- Algorithmic generation using mathematical ellipsoid models
- Based on typical bird body proportions and physics
- No real bird scans or copyrighted models used
- Created by `src/generate_species_prototypes.py`

**How We Generate Them:**
```python
# Synthetic point cloud generation
# 1. Create ellipsoid shape based on bird proportions
# 2. Apply species-specific body type (compact, crested, etc.)
# 3. Add realistic noise and variation
# 4. Scale based on typical bird measurements
# Result: Unique synthetic 3D model
```

**Rights:** Public domain (our own creation)
**License:** MIT (same as project)
**Usage:** Unlimited - research, commercial, personal

### Physical Measurements Used

The synthetic models are scaled using publicly available bird measurements:

#### Primary Sources:

1. **Cornell Lab of Ornithology - All About Birds**
   - URL: https://www.allaboutbirds.org/
   - License: Educational/Public Information
   - Data Type: Bird size, weight, identification info
   - Usage: Reference measurements for model scaling
   - Notes: Publicly available field guide information

2. **Wikipedia - Bird Species Articles**
   - Example: https://en.wikipedia.org/wiki/Northern_Cardinal
   - License: Creative Commons BY-SA 3.0 / Public Domain
   - Data Type: Scientific measurements, taxonomy
   - Usage: Typical size and weight ranges
   - Notes: Factual data (measurements) not copyrightable

3. **Field Guides (Measurements Only)**
   - Sibley Guide to Birds (measurements)
   - Peterson Field Guide to Birds (measurements)
   - Notes: Factual measurements are not copyrightable
   - We use only numerical data (size, weight), not images or descriptions

#### Example Data:

```json
"Cardinal": {
  "typical_size_cm": 22,     // From: Cornell Lab, Wikipedia
  "typical_weight_g": 45,    // From: Cornell Lab, Wikipedia
  "body_type": "crested"     // From: Visual observation (public knowledge)
}
```

**Important:** We use only **factual measurements** (which are not copyrightable), not images, artwork, or descriptive text.

---

## Species Configuration

### Source: **Public Domain Scientific Data**

**File:** `species_3d_support.json`

**Data Sources:**

1. **Common Names**
   - Source: American Ornithological Society (AOS)
   - URL: https://americanornithology.org/
   - License: Public scientific nomenclature
   - Usage: Standardized bird names

2. **Scientific Names (Taxonomy)**
   - Source: IOC World Bird List
   - URL: https://www.worldbirdnames.org/
   - License: Public domain (scientific taxonomy)
   - Usage: Species classification

3. **Physical Characteristics**
   - Sources: Cornell Lab, Wikipedia, Field observations
   - License: Public domain (factual data)
   - Usage: Size, weight, body proportions

**Rights:** Public domain (scientific facts)
**License:** CC0 / Public Domain
**Usage:** Unlimited

---

## YOLO Detection Models

### Source: **User-Provided**

**Location:** User must provide their own YOLO models

**Expected Sources:**
1. **YOLOv8 Base Models** (Ultralytics)
   - URL: https://github.com/ultralytics/ultralytics
   - License: AGPL-3.0
   - Usage: Can be used for research and commercial (with license compliance)

2. **Custom Trained Models** (User's own)
   - User trains on their own bird images
   - Rights: User owns their trained models
   - Usage: Unlimited (user's IP)

**Our System:**
- We do NOT provide pre-trained bird detection models
- User must train their own or use compatible models
- System is model-agnostic (works with any YOLO model)

---

## Camera Data

### Source: **User-Generated**

**Location:** Captured in real-time from user's cameras

**Data Types:**
- RGB images from user's USB camera
- Depth data from user's CS20 ToF camera
- Point clouds generated from user's depth data

**Rights:** User owns all captured data
**License:** User's choice
**Usage:** Unlimited (user's own data)

---

## Individual Bird Database

### Source: **User-Generated**

**Location:** `birdid.sqlite`

**Data:**
- 3D point cloud embeddings from birds visiting user's feeder
- Timestamps and statistics
- User-defined bird names and metadata

**Rights:** User owns all enrollment data
**License:** User's choice
**Usage:** Unlimited (user's own data)

---

## Software Libraries

### Dependencies and Their Licenses:

| Library | License | Usage | Link |
|---------|---------|-------|------|
| PyTorch | BSD-3-Clause | Neural networks | https://pytorch.org/ |
| OpenCV | Apache 2.0 | Computer vision | https://opencv.org/ |
| Open3D | MIT | 3D processing | http://www.open3d.org/ |
| NumPy | BSD-3-Clause | Numerical computing | https://numpy.org/ |
| Flask | BSD-3-Clause | Web server | https://flask.palletsprojects.com/ |
| Ultralytics | AGPL-3.0 | YOLO framework | https://ultralytics.com/ |

**All compatible with research and commercial use** (with license compliance)

---

## Research Data Guidelines

### For Academic Research:

✅ **Approved Uses:**
- Scientific bird behavior studies
- Individual bird identification research
- Computer vision algorithm development
- Wildlife conservation research
- Ecological monitoring studies

### Citation:

If using this system in research, please cite:

```bibtex
@software{ornimetrics2024,
  title={Ornimetrics: 3D Individual Bird Identification System},
  author={Your Name/Organization},
  year={2024},
  url={https://github.com/yourusername/Ornimetrics-AI},
  note={Open source bird detection system with 3D point cloud recognition}
}
```

### Data Sharing:

- ✅ You may share synthetic PLY prototypes
- ✅ You may share your own captured images/point clouds
- ✅ You may publish your individual bird database
- ✅ You may share anonymized statistics

---

## Commercial Use

### ✅ Permitted:

- Using the system for commercial bird feeders
- Selling products based on this technology
- Offering bird monitoring services
- Using in smart feeder products

### ⚠️ Requirements:

- Comply with Ultralytics AGPL-3.0 license (if using their YOLO models)
  - Either: Release your code as open source under AGPL-3.0
  - Or: Purchase commercial license from Ultralytics
- Other components (our code, synthetic data) are MIT licensed

### 📄 Project License:

This project is licensed under **MIT License** (except where noted)

---

## Data Privacy

### User Data:

- All bird detections happen locally on device
- No data sent to external servers (unless user configures Firebase)
- User controls all data storage and sharing
- Individual bird database stays on user's device

### Firebase (Optional):

If user enables Firebase:
- User must provide their own Firebase account
- User controls data access and privacy
- Follows user's Firebase security rules

---

## No Copyright Infringement

### We Do NOT Use:

❌ Copyrighted bird photographs
❌ Proprietary 3D scans
❌ Licensed artwork or models
❌ Restricted scientific datasets
❌ Any paid/restricted resources

### We DO Use:

✅ Synthetic procedurally-generated models
✅ Public domain scientific measurements
✅ Open source software libraries
✅ User's own captured data
✅ Our own original code

---

## Legal Compliance

### GDPR (EU):
- System processes data locally
- No personal data collected by system
- User controls all data

### CCPA (California):
- No user data sold or shared
- User owns all captured data
- Privacy by design

### Wildlife Protection:
- Non-invasive observation only
- No harm to birds
- Ethical wildlife monitoring

---

## Additional Resources

### Scientific Bird Data (Public Domain):

1. **eBird** (Cornell Lab)
   - URL: https://ebird.org/
   - License: Open data
   - Usage: Bird distribution and observation data

2. **USGS Bird Banding Lab**
   - URL: https://www.usgs.gov/centers/eesc/science/bird-banding-laboratory
   - License: US Government Public Domain
   - Usage: Bird tracking and population data

3. **Avibase - The World Bird Database**
   - URL: https://avibase.bsc-eoc.org/
   - License: Open data
   - Usage: Taxonomic information

### Computer Vision Datasets (If You Need Training Data):

1. **NABirds Dataset**
   - URL: http://dl.allaboutbirds.org/nabirds
   - License: Academic research
   - Usage: Training custom models (check license)

2. **CUB-200-2011**
   - URL: http://www.vision.caltech.edu/datasets/cub_200_2011/
   - License: Academic research
   - Usage: Training custom models (check license)

**Note:** We do NOT include these datasets. Users must obtain them separately if needed.

---

## Questions?

### Licensing Questions:
- All synthetic data: MIT License
- All our code: MIT License
- Third-party libraries: See table above

### Data Questions:
- Measurements: Public domain (scientific facts)
- PLY models: Our creation (MIT License)
- Your captures: You own them

### Research/Commercial Use:
- ✅ Safe for both (see notes above)
- Check Ultralytics license for YOLO
- All other components are permissive

---

## Summary

✅ **All data is legally sourced**
✅ **No copyright infringement**
✅ **Safe for research use**
✅ **Safe for commercial use** (with YOLO license compliance)
✅ **All sources documented**
✅ **Transparent and open**

**Last Updated:** 2024-01-17
**Version:** 1.0.0

---

## Contact

For data licensing questions or attribution requests:
- Open an issue on GitHub
- Check project README for contact info
- Review MIT License for project terms
