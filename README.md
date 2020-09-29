WARNING: This public respository is still in beta... There has been a considerable rewrite (particularly swapping ChumPy with PyTorch3D, using PyTorch etc.) since the original paper. I am bringing this up to speed in my free moments, but for now, please use at your own risk :)

# Creatures Great and SMAL
Fitting code used as part of the [Creatures Great and SMAL](https://arxiv.org/abs/1811.05804) paper.

## Installation
1. Clone the repository with submodules and enter directory
   ```
   git clone --recurse-submodules https://github.com/benjiebob/CreaturesGreatAndSMAL
   cd CreaturesGreatAndSMAL
   ```
    
2. Install dependencies, particularly [PyTorch (cuda support recommended)](https://pytorch.org/), [Pytorch3D](https://github.com/facebookresearch/pytorch3d)

3. Clone the [SMAL-ST](http://smal.is.tue.mpg.de/) project website in order to access the latest version of the SMAL deformable animal model. You should copy all of [these files](https://github.com/silviazuffi/smalst/tree/master/smpl_models) underneath a CreaturesGreatAndSMAL/data/smal directory. 

   Windows tip: If you are a Windows user, you can use these files but you'll need to edit the line endings. Try the following Powershell commands, shown here on one example:
     ```
     $path="my_smpl_00781_4_all_template_w_tex_uv_001.pkl"
     (Get-Content $path -Raw).Replace("`r`n","`n") | Set-Content $path -Force
     ```

   For more information, check out the StackOverflow answer [here](https://stackoverflow.com/questions/19127741/replace-crlf-using-powershell)

4. Download images corresponding to demo BADJA sequence
   ```
   cd data/BADJA
   wget http://mi.eng.cam.ac.uk/~bjb56/datasets/badja_extra_videos.zip
   unzip badja_extra_videos.zip

5. Inspect the directory paths in config.py and make sure they match your system.

## Sample Usage
The behaviour of this code is adaptable by using the config.py file.

### Run on a sample video from the [BADJA](https://github.com/benjiebob/BADJA) dataset.

   1. Run the python script
   ```
   python optimize_to_joints.py
   ```

### Fit to an image from [StanfordExtra](https://github.com/benjiebob/StanfordExtra) dataset.

   1. Edit the config.py file to make load a StanfordExtra image instead of a BADJA video sequence:

   ```
   #SEQUENCE_OR_IMAGE_NAME = "badja:rs_dog"
   SEQUENCE_OR_IMAGE_NAME = "stanfordextra:n02092339-Weimaraner/n02092339_748.jpg"
   ```
   2. Run the python script:
   ```
   cd smal_fitter
   python optimize_to_joints.py
   ```

### Tutorial Notes and Conventions
#### Running other BADJA/StanfordExtra sequences:
   1. Change the SEQUENCE_OR_IMAGE_NAME to "DATASET:SEQUENCE_NAME"
   2. Change the SHAPE_FAMILY # Choose from Cat (e.g. House Cat/Tiger/Lion), Canine (e.g. Dog/Wolf)
   3. Check the IMAGE_RANGE. # Frames to process from sequence. Ignored for stanford extra
   4. Check the WINDOW_SIZE. For video sequences, this is the number of frames to fit into a batch.

#### Running on your own data
The first job is to source keypoint/silhouette data for your input images. 

- Manual Annotation
   - I recommend using [LabelMe](https://github.com/wkentaro/labelme), which is fantastic software that makes annotating keypoints / silhouettes efficient.
   - Install the software, and then load the joint annotation execute
      ```
      labelme --labels labels.txt --nosortlabels
      ```
   - Next, generate the silhouette annotations
      ```
      # TODO
      ```
   - TODO: Write script to load labelme files
- StanfordExtra Joint Prediction
   - TODO: Release model for dog keypoint prediction, trained on StanfordExtra
- CreaturesGreatAndSMAL (CGAS) Joint Prediction
   - TODO: Release training/testing scripts for CreaturesGreatAndSMAL joint prediction, OJA methods etc.

### Acknowledgements
If you find this fitting code and/or BADJA dataset useful for your research, please consider citing the following paper:

```
@inproceedings{biggs2018creatures,
  title={{C}reatures great and {SMAL}: {R}ecovering the shape and motion of animals from video},
  author={Biggs, Benjamin and Roddick, Thomas and Fitzgibbon, Andrew and Cipolla, Roberto},
  booktitle={ACCV},
  year={2018}
}
```

if you make use of the limb scaling parameters, or Unity shape prior (on by default for the dog shape family) or the [StanfordExtra](https://github.com/benjiebob/StanfordExtra) dataset please cite [Who Left the Dogs Out? 3D Animal Reconstruction with Expectation Maximization in the Loop](https://arxiv.org/abs/2007.11110):

```
@inproceedings{biggs2020wldo,
  title={{W}ho left the dogs out?: {3D} animal reconstruction with expectation maximization in the loop},
  author={Biggs, Benjamin and Boyne, Oliver and Charles, James and Fitzgibbon, Andrew and Cipolla, Roberto},
  booktitle={ECCV},
  year={2020}
}
```

## Contribute
Please create a pull request or submit an issue if you would like to contribute.
