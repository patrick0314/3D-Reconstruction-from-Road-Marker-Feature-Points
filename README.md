# 3D-Reconstruction-from-Road-Marker-Feature-Points

## ☆ Downloads the project and the datasets
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
├── run.sh
└── ReadMe.md
```

## ☆ Prerequisites
* matplotlib==3.3.4
* numpy==1.24.3
* opencv==4.6.0
* opencv-contrib-python==4.7.0.72
* open3d==0.17.0
* pandas==1.5.272
* python==3.9.0
* tqdm==4.65.0

## ☆ Usage
### ◇ Run with commands
Type the command below and replace `{seq}` with your desired sequence folder name and default is `seq1`

Type the command below and replace `{threshold}` with your desired ICP threshold aand defualt is `0.47`

After running, the program generates a corresponding point cloud folder - `pcd/{seq}` and a solution text file - `pred_pose_{seq}.txt`
```
python src/main.py --seq {seq} --ICP_threshold {threshold}
```
### ◇ Run with script files
Modify the `run.sh`:
* use the command as above.
* change the `{seq}` and `{threshold` as you want.
  
After modification, type the command below.

After running, the program generates corresponding point cloud folders - `pcd/{seq}` and solution text files - `pred_pose_{seq}.txt`
```
bash run.sh
```

## ☆ Results
### ◇ Training data
Use the script file - `run.sh` to run `seq1`, `seq2` and `seq3` at once.

After running, the program generates corresponding point cloud folders - `pcd/seq1`, `pcd/seq2` and `pcd/seq3` and generate solution text files - `pred_pose_seq1.txt`, `pred_pose_seq2.txt` and `pred_pose_seq3.txt`.

Also, since these three folder has groundtruth, the corresponding Mean Error are shown on the screen.

![results](https://github.com/patrick0314/3D-Reconstruction-from-Road-Marker-Feature-Points/assets/47914151/301b1266-9c4f-4673-a9b3-ad368db58ed9)

### ◇ Testing data
Use the script file - `run.sh` to run `test1` and `test2` at once.

After running, the program generates corresponding point cloud folders - `pcd/test1` and `pcd/test2` and generate solution text files - `pred_pose_test1.txt` and `pred_pose_test2.txt`.

![results2](https://github.com/patrick0314/3D-Reconstruction-from-Road-Marker-Feature-Points/assets/47914151/48912428-4303-4234-9de4-de30207d319d)

> if one want to upload the test results onto CodaLab, move `pred_pose_test1.txt` and `pred_pose_test2.txt` into folders `test1` and `test2` respectively and rename as `pred_pose.txt`. And then, move two folders into a folder - `solution` and compress into a zip file.
