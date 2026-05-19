# CASA-Net

**CASA-Net: Context-Aware Small Object Detection Network for UAV Aerial Images**

A small object detection network based on YOLOv26s, optimized for UAV aerial scenarios.

## Key Innovations

### 1. MFEM (Multi-scale Feature Enhancement Module)

Multi-scale feature enhancement module with three parallel branches:

- **Primary Branch**: Preserves original feature information
- **Context Branch**: Aggregates cross-scale context from adjacent FPN layers
- **Attention Branch**: Spatial attention localization

Feature fusion uses channel-wise concatenation followed by 1x1 convolution refinement.

### 2. E-STAL (Enhanced Small-Target-Aware Label Assignment)

Label assignment strategy optimized for small targets:

- **Adaptive IoU Threshold**: Dynamically lowers threshold based on target scale for more reasonable positive sample matching
- **Minimum Assignment Guarantee**: Ensures tiny objects receive at least a specified number of positive samples
- **Spatial Tolerance Mechanism**: Uses relaxed spatial matching for extremely small objects

### 3. Aerial Data Augmentation

Data augmentation designed for UAV aerial imagery characteristics:

| Augmentation | Function |
|--------------|----------|
| AerialMotionBlur | Simulates motion blur from UAV flight |
| AerialShadowIllumination | Simulates shadow and illumination variations in aerial imagery |
| AerialScaleJitter | Altitude-aware scale jitter preserving small object visibility |
| AerialPerspectiveTransform | Simulates perspective distortion from UAV altitude and angle variations |


## Model Configuration

| Parameter | Description |
|-----------|-------------|
| `estal=True` | Enable E-STAL label assignment |
| `estal_min_assign=6` | Minimum positive samples for small objects |
| `estal_spatial_tolerance=0.7` | Spatial tolerance |
| `aerial_aug=True` | Enable aerial data augmentation |
| `aerial_perspective=0.5` | Aerial perspective transform probability |

