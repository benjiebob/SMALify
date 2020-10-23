3D SMAL Fitter
------

This directory contains code for optimising the SMAL/SMBLD model to provided 3D models.

It optimises shape, pose, scale to allow for the production of new shape & pose priors based on the set of animal models provided.

## Input file format

Currently, this only supports .obj files.

To avoid errors in initialisation, you may need to manually align your model so that it faces the same direction as the SMAL model. This alignment is with tail to head going in the positive x direction, and z denoting vertical height.

## Quickstart

Two options to run an optimisation:

- Single stage. Pass args into optimise.py to start a single scheme of optimisation, for example:

`python fitter_3d/optimise.py --mesh_dir example_meshes --scheme default --lr 1e-3 --nits 100`

- For a more complicated (eg multi-stage) and fine tuned optimisation, add a custom .yaml file. See example_cfg.yaml for how it must be organised. This can then be called in optimise.py using:

`python fitter_3d/optimise.py --mesh_dir example_meshes --yaml_src fitter_3d/example_cfg.yaml`

Note: Any args provided in example_cfg.yaml will overwrite args provided at the command line.

## Schemes

Five optimisation schemes (defined in trainer.py/SMALParamGroup):

- `init` - Global rotation, global translation
- `default` - Global rotation, global translation, shape parameters, joint rotations
- `shape` - `default` without joint rotations
- `pose` - `default` without shape parameters
- `deform` - Vertex deformations only



