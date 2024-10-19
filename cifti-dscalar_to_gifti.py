import numpy as np
import nibabel as nb 

# function below copied from https://nbviewer.org/github/neurohackademy/nh2020-curriculum/blob/master/we-nibabel-markiewicz/NiBabel.ipynb
def surf_data_from_cifti(data, axis, surf_name):
    assert isinstance(axis, nb.cifti2.BrainModelAxis)
    for name, data_indices, model in axis.iter_structures():  # Iterates over volumetric and surface structures
        if name == surf_name:                                 # Just looking for a surface
            data = data.T[data_indices]                       # Assume brainmodels axis is last, move it to front
            vtx_indices = model.vertex                        # Generally 1-N, except medial wall vertices
            surf_data = np.zeros((vtx_indices.max() + 1,) + data.shape[1:], dtype=data.dtype)
            surf_data[vtx_indices] = data
            return surf_data
    raise ValueError(f"No structure named {surf_name}")

# check if there's left/right structure 
def check_cifti_structures(cifti_path):
    cifti = nb.load(cifti_path)
    cortex_structure_hems_counter = 0
    cortex_structure_names = []
    for name, _, _ in cifti.header.get_axis(1).iter_structures():
        cortex_structure_names += [name]
        if name == 'CIFTI_STRUCTURE_CORTEX_LEFT':
            cortex_structure_hems_counter += 1
        elif name == 'CIFTI_STRUCTURE_CORTEX_LEFT':
            cortex_structure_hems_counter += 1
    return (cortex_structure_hems_counter == 2, cortex_structure_names)

# modified from decompose_cifti() in link from above comment to get GIFTIs only
def decompose_cifti_to_gifti(img):
    data = img.get_fdata(dtype=np.float32)
    brain_models = img.header.get_axis(1)  # Assume we know this
    return (surf_data_from_cifti(data, brain_models, "CIFTI_STRUCTURE_CORTEX_LEFT"),
            surf_data_from_cifti(data, brain_models, "CIFTI_STRUCTURE_CORTEX_RIGHT")
            )

def output_gifti(cifti_path, output_path = None):
    cifti = nb.load(cifti_path)
    left_darr, right_darr = decompose_cifti_to_gifti(cifti)

    left_gii = nb.GiftiImage(darrays=[left_darr])
    right_gii = nb.GiftiImage(darrays=[right_darr])

    if output_path is not None: 
        nb.save(left_gii, f"{output_path}.L.gii")
        nb.save(right_gii, f"{output_path}.R.gii")

    return left_gii, right_gii

###########################################################################################################################
# example usage to convert cifti dscalar to gifti

# run these lines first to check that there is left/right cortex in the cifti format
# it should return (True, ['CIFTI_STRUCTURE_CORTEX_LEFT', 'CIFTI_STRUCTURE_CORTEX_RIGHT']) 
cifti_path = 'data/sub-MSC01_networks.dscalar.nii'
check_cifti_structures(cifti_path)

# if the check passes, these two lines should work and 
# it should output the files 'sub-MSC01_networks.L.gii', 'sub-MSC01_networks.R.gii'
data_name = cifti_path.split('/')[1].split('.')[0]
left_gii, right_gii = output_gifti(cifti_path, f'{data_name}')
###########################################################################################################################