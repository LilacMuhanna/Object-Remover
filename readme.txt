
# Advanced DL Course Final Project: Digital Image Automation
Contributors:: Adi Dokhan & Lilac Muhanna

## Project Overview

This project focuses on developing automated methods for digital image editing, particularly the detection and removal of objects from images. Utilizing the COCO 2017 dataset and the FiftyOne library, we enhanced the process efficiency from data preprocessing to model training.

## Components

1. Object Detection in Images: Implementation of YOLOv5 for real-time object detection and its optimization for vehicle identification within images.
2. Object Removal from Images: Use of Mask R-CNN for accurate object segmentation followed by the DeepFill GAN for reconstructing the image areas from which objects were removed.

## Highlights

- Data Handling: Automated the data conversion from COCO's JSON to YOLO format using FiftyOne, focusing on vehicle images to refine detection relevance.
- Model Efficiency: Customized YOLOv5 to detect objects swiftly and accurately, with DeepFill GAN effectively filling in the removed areas, ensuring natural-looking images.
- User Interface: Developed a user-friendly interface that allows for easy upload and processing of images, providing instant results without user-intensive operations.

## Conclusion

Our project demonstrates significant advancements in automated image processing, proving effective in both detecting and seamlessly removing objects. This sets a robust foundation for further exploration and development of more sophisticated image editing tools.

