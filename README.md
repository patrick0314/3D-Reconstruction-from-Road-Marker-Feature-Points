# 3D-Reconstruction-from-Road-Marker-Feature-Points

## Downloads the project and the datasets
1. git clone the repo - `Breadcrumbs3D-Reconstruction-from-Road-Marker-Feature-Points` and get into the folder.
2. Download and unzip `ITRI_dataset.zip`, `ITRI_DLC.zip` and `ITRI_DLC2.zip` into the folder.
3. Move folder - `test1` and `test2` from `ITRI_DLC` into `ITRI_dataset`
4. Run `mvDLC.py` to move the `new initial pose`, `gt_pose.txt` and `localization_timestamp.txt` from `ITRI_DLC2` to `ITRI_dataset`
5. (optional) delete the zip file `ITRI_dataset.zip`, `ITRI_DLC.zip` and `ITRI_DLC2.zip` to make the folder clean
5. (optional) delete the folder `ITRI_DLC` and `ITRI_DLC2` to make the folder clean

The directory structure shows below:
```
3D-Reconstruction-from-Road-Marker-Feature-Points/
|
├── ITRI_dataset/
|     ├── camera_info/
|     ├── seq1/
|     ├── seq2/
|     ├── seq3/
|     ├── test1/
|     ├── test2/
|     └── ReadMe.md
├── slides/
├── src/
├── others/
├── mvDLC.py
└── run.sh
```
