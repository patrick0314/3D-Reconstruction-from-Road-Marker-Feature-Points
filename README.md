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

## Prerequisites
* matplotlib==3.3.4
* numpy==1.24.3
* opencv==4.6.0
* opencv-contrib-python==4.7.0.72
* open3d==0.17.0
* pandas==1.5.272
* python==3.9.0
* tqdm==4.65.0

## Usage
### Run with commands
Type the command below and replace `{seq}` with your desired sequence folder name and default is `seq1`
Type the command below and replace `{threshold}` with your desired ICP threshold aand defualt is `0.47`
```
python src/main.py --seq {seq} --ICP_threshold {threshold}
```
### Run with script files
Type the command below and replace `{seq}` with your desired sequence folder name like `seq1`, `seq2` or `seq3`.
```
python src/main.py --seq {seq}
```
