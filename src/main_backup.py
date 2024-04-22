#!/usr/bin/env python3
import argparse
import matplotlib
matplotlib.use('TkAgg')  # Use the Tkinter backend for interactive plots

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np
import os
import torch
from objRemove import ObjectRemove
from models2.deepFill import Generator
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
import torch
import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath

from PIL import Image, ImageOps
model_path = r'C:\\Users\\Adi\\Desktop\\fourthyear\\DL-Proj\\object-remove-main\\src\\models2\\best.pt'


# Assuming 'best.pt' is your custom trained YOLOv5 model.
model_custom = torch.hub.load(r'C:\Users\Adi\Desktop\fourthyear\DL-Proj\object-remove-main\yolov5', 'custom', path=model_path, source='local', force_reload=True)
model_custom.eval()  # Set the model to inference mode

# Function to get detections
def get_detections(image_path, model, conf=0.5):
    results = model(image_path)
    results = results.pandas().xyxy[0]  # Results as DataFrame
    detections = results[results['confidence'] > conf]
    return detections[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()

# def resize_image(image_path, target_size=(512, 680)):
#     """Resize the image maintaining aspect ratio."""
#     image = Image.open(image_path)
#     image = image.resize(target_size, Image.ANTIALIAS)
#     return image
# from PIL import Image, ImageOps

# Global variable to store the selected detection
selected_detection = None

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check which bounding box was clicked and proceed accordingly
        for i, bbox in enumerate(params['detections']):
            xmin, ymin, xmax, ymax = bbox
            if xmin <= x <= xmax and ymin <= y <= ymax:
                print(f"Car {i+1} selected for removal")
                params['selected_bbox'] = bbox
                cv2.destroyAllWindows()
                break

def display_and_select(image_path, detections):
    img = cv2.imread(image_path)
    for i, (xmin, ymin, xmax, ymax) in enumerate(detections):
        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
        cv2.putText(img, f"Car {i+1}", (int(xmin), int(ymin-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Dictionary to store function parameters and results
    params = {'detections': detections, 'selected_bbox': None}
    
    cv2.imshow("Select a car to remove by clicking on it", img)
    cv2.setMouseCallback("Select a car to remove by clicking on it", click_event, params)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return params['selected_bbox']

# Assuming `detections` is a list of detections where each detection is [xmin, ymin, xmax, ymax]


def resize_image(image_path, target_size=(512, 680), output_format='JPEG'):
    """Resize the image maintaining aspect ratio and save to the specified format in a given directory."""
    image = Image.open(image_path)
    image = ImageOps.contain(image, target_size, method=Image.Resampling.LANCZOS)
    
    # Directory where the resized image will be saved
    output_dir = "C:/Users/Adi/Desktop/fourthyear/DL-Proj/object-remove-main/resized"

    # Extract the filename and extension
    filename = os.path.basename(image_path)
    filename_without_ext = os.path.splitext(filename)[0]
    
    # Specify an output path within the output_dir with a file extension matching the output format
    output_path = os.path.join(output_dir, f"{filename_without_ext}_resized.{output_format.lower()}")
    
    # Save the image with the specified format
    image.save(output_path, format=output_format)
    return output_path

##################################
# Get image path from command line
##################################
parser = argparse.ArgumentParser()
parser.add_argument("image")
args = parser.parse_args()
image_path = args.image
resized_image = resize_image(image_path)
resized_image_path = resize_image(image_path)

image_path = resized_image_path

# Get detections for the image
detections = get_detections(image_path, model_custom)
print("Display and select")
# selected_detections=display_and_select(detections,image_path)
selected_bbox = display_and_select(image_path, detections)

# if selected_bbox:
#     # Process the selected_bbox with your ObjectRemove
#     model = ObjectRemove(image_path=image_path, detections=[selected_bbox])
#     output = model.run()
#     # Display or save the output image as needed
# else:
#     print("No car was selected for removal.")
# # Load the image
# image = cv2.imread(image_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

# # Draw each detection on the image
# for detection in adjusted_detections:  # Assuming detections are [xmin, ymin, xmax, ymax]
#     start_point = (int(detection[0]), int(detection[1]))  # Top left corner
#     end_point = (int(detection[2]), int(detection[3]))  # Bottom right corner
#     color = (255, 0, 0)  # Blue color in RGB
#     thickness = 2  # Line thickness
#     image = cv2.rectangle(image, start_point, end_point, color, thickness)

# # Convert array to Image for display in Jupyter Notebook
# display_image = Image.fromarray(image)

# # Display the image with detections
# display_image.show()
######################################################
# Creating Mask-RCNN model and load pretrained weights
######################################################
print("Creating rcnn model")
weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
transforms = weights.transforms()
rcnn = maskrcnn_resnet50_fpn(weights=weights, progress=False)
rcnn.eval()

#########################
# Create inpainting model
#########################
deepfill_weights_path = 'C:/Users/Adi/Desktop/fourthyear/DL-Proj/object-remove-main/src/models2/states_pt_places2.pth' 
print('Creating deepfill model')
deepfill = Generator(checkpoint=deepfill_weights_path, return_flow=True)

######################
# Create ObjectRemoval
######################
model = ObjectRemove(segmentModel=rcnn, rcnn_transforms=transforms, inpaintModel=deepfill, image_path=image_path, detections=[selected_bbox])  # Adjust ObjectRemove to accept detections

#####
# Run
#####
output = model.run()

# # Display Results
# plt.figure(figsize=(15, 5))
# plt.subplot(1, 3, 1)
# plt.imshow(cv2.cvtColor(cv2.imread(args.image_path), cv2.COLOR_BGR2RGB))
# plt.title('Original Image')
# plt.subplot(1, 3, 2)
# plt.imshow(model.image_masked)
# plt.title('Masked Image')
# plt.subplot(1, 3, 3)
# plt.imshow(output)
# plt.title('Output Image')
# plt.show()
#################
# Display results
#################
img = cv2.cvtColor(model.image_orig[0].permute(1,2,0).numpy(), cv2.COLOR_RGB2BGR)
boxed = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig, axs = plt.subplots(1, 3, figsize=(20, 5))
axs[0].imshow(boxed)
axs[0].set_title('Original Image with Detections')
axs[1].imshow(model.image_masked.permute(1,2,0).detach().numpy())
axs[1].set_title('Masked Image')
axs[2].imshow(output)
axs[2].set_title('Inpainted Image')


fig.savefig('output_figure.png')
import os
print("Current Working Directory:", os.getcwd())


