# Creatures Great and SMAL
Fitting code used as part of the Creatures Great and SMAL paper.

## Installation
1. Clone the repository with submodules and enter directory
   ```
   git clone --recurse-submodules https://github.com/benjiebob/CreaturesGreatAndSMAL
   cd CreaturesGreatAndSMAL
    
2. Install the Neural Mesh Renderer submodule (which originated from my [fork](https://github.com/benjiebob/neural_renderer)), which includes a render_points function

   ```
   pip install -e neural_renderer
   ```

3. Download texture map (from smal/dog_texture.pkl) and a version of SMAL 2017 converted to NumPy (smal_CVPR2017_np.pkl) from [my Google Drive](https://drive.google.com/open?id=1gPwA_tl1qrKiUkveE8PTsEOEMHtTw8br) and place under the smal folder

4. Install dependencies, particularly [PyTorch (with cuda support)](https://pytorch.org/)

5. Create a checkpoints directory

6. Run the python3 script
   ```
   cd smal_fitter
   python optimize_to_joints.py
   ```

### Acknowledgements
If you find this fitting code useful for your research, please consider citing the following paper:

```
@inproceedings{biggs2018creatures,
  title={{C}reatures great and {SMAL}: {R}ecovering the shape and motion of animals from video},
  author={Biggs, Benjamin and Roddick, Thomas and Fitzgibbon, Andrew and Cipolla, Roberto},
  booktitle={ACCV},
  year={2018}
}
```
