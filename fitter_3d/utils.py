import os
import torch

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes

from matplotlib.tri import Triangulation

def try_mkdir(loc):
	if not os.path.isdir(loc):
		os.mkdir(loc)


def try_mkdirs(locs):
	for loc in locs: try_mkdir(loc)

def equal_3d_axes(ax, X, Y, Z, zoom=1.0):
	"""Sets all axes to same lengthscale through trick found here:
	https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to"""

	xmax, xmin, ymax, ymin, zmax, zmin = X.max(), X.min(), Y.max(), Y.min(), Z.max(), Z.min()

	max_range = np.array([xmax - xmin, ymax - ymin, zmax - zmin]).max() / (2.0 * zoom)

	mid_x = (xmax + xmin) * 0.5
	mid_y = (ymax + ymin) * 0.5
	mid_z = (zmax + zmin) * 0.5
	ax.set_xlim(mid_x - max_range, mid_x + max_range)
	ax.set_ylim(mid_y - max_range, mid_y + max_range)
	ax.set_zlim(mid_z - max_range, mid_z + max_range)


def plot_trisurfs(ax, verts, faces, change_lims=False, color="darkcyan",
				zoom=1.5, n_meshes=1, alpha=1.0):
	"""
	"""

	trisurfs = []

	for n in range(n_meshes):
		points = verts[n].cpu().detach().numpy()
		faces = faces[n].cpu().detach().numpy()

		X, Y, Z = np.rollaxis(points, -1)
		tri = Triangulation(X, Y, triangles=faces).triangles

		trisurf_shade = ax.plot_trisurf(X, Y, Z, triangles=tri, alpha=alpha, color=color,
										shade=True)  # shade entire mesh
		trisurfs += [trisurf_shade]

	if change_lims: equal_3d_axes(ax, X, Y, Z, zoom=zoom)

	return trisurfs


def plot_mesh(ax, mesh: Meshes, label="", colour="blue", equalize=True, zoom=1.5, alpha=1.0):
	"""Given a PyTorch Meshes object, plot the mesh on a 3D axis

	:param equalize: Flag to match axes to be the same scale
	:param zoom: Zoom factor on equalised axes
	"""

	verts = mesh.verts_padded()
	faces = mesh.faces_padded()
	trisurfs = plot_trisurfs(ax, verts, faces, color=colour, change_lims=equalize, zoom=zoom,
			alpha = alpha)
	ax.plot([], [], color=colour, label=label)

	return trisurfs


def plot_meshes(target_meshes, src_meshes, mesh_names=[], title="", figtitle="", out_dir="static_fits_output/pointclouds"):
	"""Plot and save fig of point clouds, with 3 sublpots side by side:
	[target mesh, src_mesh, both]"""

	for n in range(len(target_meshes)):
		fig = plt.figure(figsize=(15, 5))
		axes = [fig.add_subplot(int(f"13{n}"), projection="3d") for n in range(1, 4)]

		for ax in axes:
			ax.set_xlabel('x')
			ax.set_ylabel('y')
			ax.set_zlabel('z')

		colours = ["green", "blue"]
		labels = ["target", "SMAL"]
		for i, mesh in enumerate([target_meshes[n], src_meshes[n]]):
			for j, ax in enumerate([axes[1 + i == 1], axes[2]]):
				plot_mesh(ax, mesh, colour=colours[i], label=labels[i], alpha=[1, 0.5][j])

		fig.suptitle(figtitle)
		for ax in axes:
			ax.legend()

		if not mesh_names:
			name = n
		else:
			name = mesh_names[n]

		try_mkdir(out_dir)

		plt.savefig(
			f"{out_dir}/{name} - {title}.png")
		plt.close(fig)


def plot_pointcloud(ax, mesh, label="", colour="blue",
					equalize=True, zoom=1.5):
	"""Given a Meshes object, plots the mesh on ax (ax must have projection=3d).

	equalize = adjust axis limits such that all axes are equal"""

	verts = mesh.verts_packed()
	x, y, z = verts.clone().detach().cpu().unbind(1)
	s = ax.scatter3D(x, y, z, c=colour, label=label, alpha=0.3)

	if equalize:
		equal_3d_axes(ax, x, y, z, zoom=zoom)

	return s, (x, y, z)


def plot_pointclouds(target_meshes, src_meshes, mesh_names=[], title="", figtitle="", out_dir="static_fits_output/pointclouds"):
	"""Plot and save fig of point clouds, with 3 figs side by side:
	[target mesh, src_mesh, both]"""

	for n in range(len(target_meshes)):
		fig = plt.figure(figsize=(15, 5))
		axes = [fig.add_subplot(int(f"13{n}"), projection="3d") for n in range(1, 4)]

		for ax in axes:
			ax.set_xlabel('x')
			ax.set_ylabel('y')
			ax.set_zlabel('z')

		colours = ["green", "blue"]
		labels = ["target", "SMAL"]
		for i, mesh in enumerate([target_meshes[n], src_meshes[n]]):
			for ax in [axes[1 + i == 1], axes[2]]:
				plot_pointcloud(ax, mesh, colour=colours[i], label=labels[i])

		fig.suptitle(figtitle)
		for ax in axes:
			ax.legend()

		if not mesh_names:
			name = n
		else:
			name = mesh_names[n]

		try_mkdir(out_dir)

		plt.savefig(
			f"{out_dir}/{name} - {title}.png")
		plt.close(fig)


def cartesian_rotation(dim="x", rot=0):
	"""Given a cartesian direction of rotation, and a rotation in radians, returns Tensor rotation matrix"""

	i = "xyz".find(dim)
	R = torch.eye(3)
	if rot != 0:
		j, k = (i + 1) % 3, (i + 2) % 3  # other two of cyclic triplet
		R[j, j] = R[k, k] = np.cos(rot)
		R[j, k] = - np.sin(rot)
		R[k, j] = np.sin(rot)

	return R


def stack_as_batch(tensor: torch.Tensor, n_repeats=1, dim=0) -> torch.Tensor:
	"""Inserts new dim dimension, and stacks tensor n times along that dimension"""
	res = tensor.unsqueeze(dim)
	repeats = [1] * res.ndim
	repeats[dim] = n_repeats # repeat across target dimension
	res = res.repeat(*repeats)
	return res


def animator(ax):
	"""Wrapper used for animating meshes.
	- Clears all current trisurfs
	- Runs func, which returns new meshes
	- Plot these meshes.

	func must contain at least verts, faces"""
	def wrapper(func):
		# aux is wrapper function sent to wrap around existing anim
		def aux(frame):
			[child.remove() for child in ax.get_children() if isinstance(child, Poly3DCollection)]
			kwargs = func(frame)
			assert "mesh" in kwargs, "anim function must return 'mesh' object"
			plot_mesh(ax, **kwargs)
		return aux
	return wrapper


def save_animation(fig, func, n_frames, fmt="gif", fps=15, title="output", callback=True, **kwargs):
	"""Save matplotlib animation."""

	arap_utils.save_animation(fig, func, n_frames, fmt=fmt, fps=fps, title=title, callback=callback, **kwargs )


def load_meshes(mesh_dir: str, sorting = lambda arr: arr, n_meshes=None, frame_step=1, device="cuda:0"):
	"""Given a dir of .obj files of Unity meshes, loads all and returns mesh names, and meshes as Mesh object.

	:param mesh_dir: Location of directory of .obj files
	:param sorting: Optional lambda function to sort listdir
	:param n_meshes: Optional slice of first n_meshes in sorted dir
	:param frame_step: For animations, optional step through sorted dir
	:param device: torch device

	:returns mesh_names: list of all names of mesh files loaded
	:returns target_meshes: PyTorch3D Meshes object of all loaded meshes
	"""

	# load all meshes
	mesh_names = []
	all_verts, all_faces_idx = [], []

	file_list = [f for f in os.listdir(mesh_dir) if ".obj" in f]

	obj_list = sorting(file_list)[::frame_step] # get sorted list of obj files, applyinfg frame step
	if n_meshes is not None: obj_list = obj_list[:n_meshes]


	for obj_file in obj_list:
		mesh_names.append(obj_file[:-4]) # Get name of mesh
		target_obj = os.path.join(mesh_dir, obj_file)
		verts, faces, aux = load_obj(target_obj, load_textures=False) # Load mesh with no textures
		faces_idx = faces.verts_idx.to \
			(device) # faces is an object which contains the following LongTensors: verts_idx, normals_idx and textures_idx
		verts = verts.to(device) # verts is a FloatTensor of shape (V, 3) where V is the number of vertices in the mesh

		# Center and scale for normalisation purposes
		centre = verts.mean(0)
		verts = verts - centre
		scale = max(verts.abs().max(0)[0])
		verts = verts /scale

		# ROTATE TARGET MESH TO GET IN DESIRED DIRECTION - UNCOMMENT IF USING
		# R1 = cartesian_rotation("z", np.pi/2).to(device)
		# R2 = cartesian_rotation("y", np.pi/2).to(device)
		# verts = torch.mm(verts, R1.T)
		# verts = torch.mm(verts, R2.T)

		all_verts.append(verts), all_faces_idx.append(faces_idx)

	print(f"{len(all_verts)} target meshes loaded.")

	target_meshes = Meshes(verts=all_verts, faces=all_faces_idx) # All loaded target meshes together

	return mesh_names, target_meshes
