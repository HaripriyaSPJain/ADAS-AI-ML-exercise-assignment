# **Sensor Fusion for Autonomous Vehicles**

## **1. Data Processing**
The dataset used in this project is from the **KITTI dataset**, specifically the files:
- [2011_09_26_drive_0001_sync.zip](https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0001/2011_09_26_drive_0001_sync.zip)
- [2011_09_26_drive_0001_extract.zip](https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0001/2011_09_26_drive_0001_extract.zip)
- [2011_09_26_calib.zip](https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_calib.zip)

### **Data Loading & Processing Steps:**
1. **Camera Data:** Images are loaded from `image_02` (left camera) using OpenCV (`cv2.imread`).
2. **LiDAR Data:** Point cloud data is loaded from `.bin` files using `numpy.fromfile`, reshaped to extract x, y, z, and intensity.
3. **Calibration Data:** The `calib_velo_to_cam.txt` file is parsed to extract rotation (R) and translation (T) matrices, forming a **3x4 transformation matrix**.

---
## **2. Sensor Fusion Techniques Applied**
Sensor fusion combines **LiDAR and camera data** for object detection and tracking.

- **LiDAR-to-Camera Projection:**
  - LiDAR points are transformed into the camera frame using:
    \[ P_{camera} = R \cdot P_{lidar} + T \]
  - The transformation matrix extracted from `calib_velo_to_cam.txt` is used.

- **Bounding Box Overlay:**
  - **YOLOv5** is used for real-time object detection on the camera images.
  - The detected bounding boxes are **aligned** with projected LiDAR points.

- **Collision Risk Assessment:**
  - If an object is closer than **10m** and moving faster than **5m/s**, a **collision warning** is triggered.

---
## **3. Integration of Camera and LiDAR Data**
1. **Extract YOLOv5 bounding boxes from images.**
2. **Transform LiDAR points to the camera frame using calibration data.**
3. **Overlay LiDAR points on the image to match detected objects.**
4. **Track moving objects using DeepSORT for speed estimation.**

```python
# Convert LiDAR points to camera coordinates
camera_coords = lidar_to_camera(points, calib_matrix)

# Display image with projected LiDAR points and bounding boxes
plt.imshow(img)
for _, row in object_vectors.iterrows():
    plt.gca().add_patch(plt.Rectangle((row['xmin'], row['ymin']),
                                      row['xmax'] - row['xmin'],
                                      row['ymax'] - row['ymin'],
                                      edgecolor='red', linewidth=2, fill=False))
plt.scatter(camera_coords[:, 0], camera_coords[:, 1], c='b', s=2, alpha=1)
plt.show()
```

---
## **4. Collision Avoidance Algorithm and Results**
The algorithm checks for objects that are **too close and moving fast**:

```python
def collision_check(objects, threshold_distance=10, threshold_speed=5):
    warnings = []
    for obj in objects:
        if obj['distance'] < threshold_distance and obj['speed'] > threshold_speed:
            warnings.append("Collision Warning: Trigger Braking!")
    return warnings
```

**Example Output:**
```
["Collision Warning: Trigger Braking!"]
```

---
## **5. Visualizations**
- **Object Detection Results:** Bounding boxes overlaid on images.
- **LiDAR-Camera Fusion:** LiDAR points projected onto detected objects.
- **Collision Detection:** Highlighted high-risk objects.

**Saved Outputs:**
- `fusion_result.png` → Sensor fusion result.
- `lidar_output.png` → 3D LiDAR visualization.

---
## **6. References and Source Links**
- **KITTI Dataset:** [http://www.cvlibs.net/datasets/kitti/](http://www.cvlibs.net/datasets/kitti/)
- **YOLOv5:** [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
- **DeepSORT Tracker:** [https://github.com/levan92/deep_sort_realtime](https://github.com/levan92/deep_sort_realtime)
- **Open3D Library:** [http://www.open3d.org/](http://www.open3d.org/)

---
### **How to Run the Code**
1. Install dependencies:
   ```bash
   pip install open3d numpy opencv-python torch torchvision pykitti
   ```
2. Mount Google Drive and set file paths.
3. Run the script in a Jupyter Notebook or Colab.

**Authors:** *Your Name*

