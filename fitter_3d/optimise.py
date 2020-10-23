import torch
import argparse
import yaml
import os

# add correct paths
import sys

if os.getcwd().endswith('fitter_3d'):  # if starting in fitter_3d dir
	os.chdir('../')
	sys.path.append('fitter_3d')

from fitter_3d.utils import load_meshes
from fitter_3d.trainer import Stage, StageManager, SMALParamGroup, SMAL3DFitter

parser = argparse.ArgumentParser()

parser.add_argument('--results_dir', type=str, default='fit3d_results', help="Directory in which results are stored")

# Mesh loading arguments
parser.add_argument('--mesh_dir', type=str, default='fitter_3d/example_meshes',
					help="Directory (relative to SMALify) in which meshes are stored")
parser.add_argument('--frame_step', type=int, default=1,
					help="If directory is a sequence of animated frames, only take every nth frame")

# SMAL args
parser.add_argument('--shape_family_id', type=int, default=-1,
					help="Shape family to use for optimisation (-1 to use default SMAL mesh)")

# yaml src
parser.add_argument('--yaml_src', type=str, default=None, help="YAML source for experimental set-up")

# optimisation scheme to be used if .yaml not found
parser.add_argument('--scheme', type=str, default='default',
					choices=list(SMALParamGroup.param_map.keys()),
					help="Optimisation scheme")
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--nits', type=int, default=100)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main(args):
	# try to load yaml
	yaml_loaded = False
	if args.yaml_src is not None:
		try:
			with open(args.yaml_src) as infile:
				yaml_cfg = yaml.load(infile, Loader=yaml.FullLoader)
		except FileNotFoundError:
			raise FileNotFoundError(
				"No YAML file found at {args.yaml_src}. Make sure this is relative to the SMALify directory.")

		yaml_loaded = True
		stage_options = yaml_cfg['stages']

		# overwrite any input args from yaml
		for arg, val in yaml_cfg['args'].items():
			setattr(args, arg, val)

	mesh_names, target_meshes = load_meshes(mesh_dir=args.mesh_dir, frame_step=args.frame_step,
											device=device)

	n_batch = len(target_meshes)
	os.makedirs(args.results_dir, exist_ok=True)
	manager = StageManager(out_dir=args.results_dir, labels=mesh_names)

	smal_model = SMAL3DFitter(batch_size=n_batch,
							  device=device, shape_family=args.shape_family_id)  # use no shape family

	stage_kwargs = dict(target_meshes=target_meshes, smal_3d_fitter=smal_model,
						out_dir=args.results_dir, device=device,
						mesh_names=mesh_names)  # standard Stage kwargs

	# if provided, load stages from YAML
	if yaml_loaded:
		for stage_name, kwargs in stage_options.items():
			stage = Stage(name=stage_name, **kwargs, **stage_kwargs)
			manager.add_stage(stage)

	# otherwise, load from arguments
	else:
		print("No YAML provided. Loading from system args. ")
		stage = Stage(scheme=args.scheme, nits=args.nits, lr=args.lr,
					  **stage_kwargs)
		manager.add_stage(stage)

	manager.run()
	manager.plot_losses('losses')  # plot to results file


if __name__ == "__main__":
	args = parser.parse_args()
	main(args)
