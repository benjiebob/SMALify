WARNING: This respository is still in beta! I am bringing this up to speed in my free moments, but for now, please use at your own risk :)

# SMALify
This repository contains a implementation for performing 3D animal (quadruped) reconstruction from a monocular image or video. The system adapts the pose (limb positions) and shape (animal type/height/weight) parameters for the SMAL deformable quadruped model, as well as camera parameters until the projected SMAL model aligns with 2D keypoints and silhouette segmentations extracted from the input frame(s).

The code can be thought of as a modernization of the fitting code used in [Creatures Great and SMAL](https://arxiv.org/abs/1811.05804) paper; Chainer/ChumPy has been replaced with PyTorch, OpenDR replaced with PyTorch3D etc. However, I have also included some recent innovations from the [Who Left the Dogs Out?](https://arxiv.org/abs/2007.11110) such as the inclusion of limb scaling parameters, and an improved shape prior.

The aim of this repository is to provide code for users to derive full 3D reconstructions for their quadruped animal image sequences. My great hope is that this work can be of some benefit to those working closely to animal science; I'd be delighted to hear from you if this is the case!

## Installation
1. Clone the repository **with submodules** and enter directory
   ```
   git clone --recurse-submodules https://github.com/benjiebob/SMALify
   cd SMALify
   ```
   Note: If you don't clone with submodules you won't get the sample data from BADJA/StanfordExtra.
    
2. Install dependencies, particularly [PyTorch (cuda support recommended)](https://pytorch.org/), [Pytorch3D](https://github.com/facebookresearch/pytorch3d)

3. Clone the [SMALST](https://github.com/silviazuffi/smalst) project website in order to access the latest version of the SMAL deformable animal model. You should copy all of [these files](https://github.com/silviazuffi/smalst/tree/master/smpl_models) underneath a SMALify/data/smal directory. 

   Windows tip: If you are a Windows user, you can use these files but you'll need to edit the line endings. Try the following Powershell commands, shown here on one example:
     ```
     $path="my_smpl_00781_4_all_template_w_tex_uv_001.pkl"
     (Get-Content $path -Raw).Replace("`r`n","`n") | Set-Content $path -Force
     ```

   For more information, check out the StackOverflow answer [here](https://stackoverflow.com/questions/19127741/replace-crlf-using-powershell)

4. Download [SMPL](https://smpl.is.tue.mpg.de/) and create a smpl_webuser directory underneath SMALify/smal_model

5. Download images corresponding to demo BADJA sequence
   ```
   cd data/BADJA
   wget http://mi.eng.cam.ac.uk/~bjb56/datasets/badja_extra_videos.zip
   unzip badja_extra_videos.zip
   ```

6. Inspect the directory paths in config.py and make sure they match your system.

## QuickStart: Running the Fitter

- Run on a sample video from the [BADJA](https://github.com/benjiebob/BADJA) dataset.
   - Run the python script
      ```
      python optimize_to_joints.py
      ```
   - Inspect the SMALify/checkpoint directory to inspect the progress of the fitter. 
      - The output files stX_epY means, stage X and iteration Y of the fitter.
      - The final output is named st10_ep0 by (a slightly lazy) convention. 
      - The other files have the following meaning

         | Extension  | Explanation |
         | ------------- | ------------- |
         | .png  | Image Visualization  |
         | .ply  | Mesh file, can be viewed in e.g. [MeshLab](https://www.meshlab.net/)  |
         | .pkl  | Pickle file, contains the latest model/camera parameters |

   - Optionally, generate a video visualization
      - Set CHECKPOINT_NAME in config.py to be the name of the output directory in CreaturesGreatAndSMAL/checkpoints
      - Run the video generation script, which exports the video to CreaturesGreatAndSMAL/exported
         ```
         python generate_video.py
         ```
- Fit to an image from [StanfordExtra](https://github.com/benjiebob/StanfordExtra) dataset.
   - Edit the config.py file to make load a StanfordExtra image instead of a BADJA video sequence:
      ```
      #SEQUENCE_OR_IMAGE_NAME = "badja:rs_dog"
      SEQUENCE_OR_IMAGE_NAME = "stanfordextra:n02092339-Weimaraner/n02092339_748.jpg"
      ```
   - Run the python script:
      ```
      cd smal_fitter
      python optimize_to_joints.py
      ```

## Running on alternative data
### Alternative BADJA/StanfordExtra sequences:
- Open the config.py file and make the following changes

   | Config Setting  | Explanation | Example |
   | ------------- | ------------- | ------------- |
   | SEQUENCE_OR_IMAGE_NAME  | Used to refer to your sequence/image  | badja:rs_dog
   | SHAPE_FAMILY            | Choose from 0: Cat, 1: Canine (e.g. Dog), 2: Equine (e.g. Horse), 3: Bovine (e.g. Cow), 4: Hippo |  1 |
   | IMAGE_RANGE  | Number of frames to process from the sequence. Ignored for StanfordExtra.  | [1,2,3] or range(0, 10) |
   | WINDOW_SIZE  | For video sequences, the number of frames to fit into a batch. Alter depending on GPU capacity. | 10 |
   
### Running on your own data
The first job is to generate keypoint/silhouette data for your input image(s). 

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
- Automatic Dog Joint Predictor
   - TODO: Release model for dog keypoint prediction, trained on StanfordExtra.
- Automatic Quadruped Joint Predictor
   - TODO: Release training/testing scripts for CreaturesGreatAndSMAL joint prediction, trained on synthetic SMAL renders, OJA methods etc.
## Improving performance and general tips and tricks
- For some sequences, it may be necessary to fiddle with the weights applied to each part of the loss function. By default the following losses are used:

| Loss Component  | Explanation | Tips
| ------------- | ------------- | ------------- |
| 2D Keypoint Reprojection  | Project the SMAL model with latest parameters and compare projected joints to input 2D keypoints | If your model limbs don't match well with the input keypoints after fitting, it may be worth increasing this. |
| 3D Shape Prior | Used to constrain the 3D shapes to be 'animal-like'. Note that (unlike equivalent human approaches that use mocap etc.) only artist data is used for this. | If your reconstructed animals don't look like animals, try increasing this. |
| 3D Pose Prior  | Used to contains the 3D poses to be anatomically plausible. | If your reconstructed animals have limb configurations which are very unreasonable, e.g. legs in strange places, try increasing this. |
| 2D Silhouette Loss  | Project the SMAL model with latest parameters and compare rendered silhouette to input 2D silhouette. | If the shape of your reconstructed animal doesn't match well (e.g. maybe it's too thin?), try increasing this. |

Note that to avoid poor local minima, the optimization proceeds over multiple stages and the weights vary at each stage. For example, only once an approximate camera location has been found should there be 2D joint loss, and only once an approximate set of limb positions have been found should there be a 2D silhouette loss.

## Acknowledgements
This repository owes a great deal to the following works and authors:
- [SMAL](http://smal.is.tue.mpg.de/); Zuffi et al. designed the SMAL deformable quadruped template model and have been wonderful for providing advice throughout my animal reconstruction PhD journey.
- [SMPLify](http://smplify.is.tue.mpg.de/); Bogo et al. provided the basis for our original ChumPY implementation and inspired the name of this repo.
- [SMALST](https://github.com/silviazuffi/smalst); Zuffi et al. provided a PyTorch implementations of the SMAL skinning functions which have been used here.

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

## Licensing
(c) Benjamin Biggs, Andrew Fitzgibbon and Roberto Cipolla. Department of Engineering, University of Cambridge 2020

By downloading this code, you agree to the [Creative Commons Attribution 3.0 International license](https://creativecommons.org/licenses/by/3.0/). This license allows users to use, share and adapt the code, so long as credit is given to the authors (e.g. by citation).

THIS SOFTWARE AND ANNOTATIONS ARE PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
