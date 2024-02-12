## Please decompress the datasets.zip here!

## 1. Dataset description
- **IDRiD**

The [IDRiD dataset](https://doi.org/10.3390/data3030025) is provided by 2018 ISBI grand challenge on diabetic retinopathy segmentation and grading, which consists of 81 color fundus images and pixel-level annotations of four types of lesions. Of these 81 images, all images contain EX and MA, a set of 80 images contain HE, and 40 images contain SE. The images have a resolution of 4288×2848 pixels. The dataset is split to 54 training samples and 27 test samples by the organizer.

## 2. Dataset structure
```bash
.
├── IDRiD
│    ├── image
│    │    ├── train
│    │    │    └── *.jpg
│    │    └── test
│    │         └── *.jpg
│    └── label
│         ├── train
│         │    ├── EX
│         │    │    └── *.tif
│         │    ├── HE
│         │    │    └── *.tif
│         │    ├── MA
│         │    │    └── *.tif
│         │    └── SE
│         │         └── *.tif
│         └── test
│              └── ...
│

```
