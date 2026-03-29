# Retail Shelf Vision Service

Production-grade computer vision system for retail shelf condition classification using deep learning and FastAPI.

---

## Overview

This project demonstrates an end-to-end ML system for analyzing retail shelf images and classifying them into business-relevant conditions such as stock availability and display quality.

It includes:
- PyTorch-based model training (ResNet transfer learning)
- Config-driven training pipeline
- REST API for real-time inference
- Batch image processing pipeline
- Dockerized deployment
- CI pipeline for automated testing

---

## Problem Statement

Retail execution teams need automated ways to monitor:
- stock availability
- shelf quality
- display compliance

This system classifies shelf images into actionable categories.

---

## Classes

### 1. Good Display
Shelf is fully stocked, well-aligned, and front-faced.

**Examples:**
- products evenly distributed
- no visible gaps
- labels facing forward

---

### 2. Low Stock
Shelf has products but noticeable gaps or low fill rate.

**Examples:**
- partially empty rows
- uneven product distribution
- visible empty sections

---

### 3. Empty Shelf
Shelf is completely or almost completely empty.

**Examples:**
- no products present
- metal rack visible
- full row empty

---

### 4. Misplaced Product
Products are present but incorrectly arranged.

**Examples:**
- wrong product in section
- mixed brands in same slot
- inconsistent arrangement

---

### 5. Poor Visibility
Image quality or visibility is degraded.

**Examples:**
- blurry image
- low lighting
- occluded shelf view

---

## Example Images

### Good Display
![Good Display](assets/examples/good_display.jpg)

### Low Stock
![Low Stock](assets/examples/low_stock.jpg)

### Empty Shelf
![Empty Shelf](assets/examples/empty_shelf.jpg)

### Misplaced Product
![Misplaced Product](assets/examples/misplaced_product.jpg)

### Poor Visibility
![Poor Visibility](assets/examples/poor_visibility.jpg)


## Dataset Structure

```text
data/raw/
├── train/
├── val/
└── test/
    ├── good_display/
    ├── low_stock/
    ├── empty_shelf/
    ├── misplaced_product/
    └── poor_visibility/