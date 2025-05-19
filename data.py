import joblib, json
import numpy as np
import torch
from functools import partial
from tqdm import tqdm
tqdm = partial(tqdm, position=0, leave=True)

from sklearn.preprocessing import OneHotEncoder
from transforms import calc_rot_mat
from pymatgen.core.structure import Structure


def RICR_represent(dataframe, max_elms=3, max_sites=20, return_Nsites=False):

    import warnings
    warnings.filterwarnings("ignore")
    

    elm_str = joblib.load('data/element.pkl')
    elm_onehot = np.arange(1, len(elm_str)+1)[:,np.newaxis]
    elm_onehot = OneHotEncoder().fit_transform(elm_onehot).toarray()
    

    with open('data/atom_init.json') as f:
        elm_prop = json.load(f)
    elm_prop = {int(key): value for key, value in elm_prop.items()}

    RICR = []
    if return_Nsites:
        Nsites = []
    op = tqdm(dataframe.index)
    rotation_matrices = []
    for idx in op:
        op.set_description(' RICR ...')
        
        crystal = Structure.from_str(dataframe['cif'][idx],fmt="cif")
        

        elm, elm_idx = np.unique(crystal.atomic_numbers, return_index=True)
        site_elm = np.array(crystal.atomic_numbers)
        elm = site_elm[np.sort(elm_idx)]
        ELM = np.zeros((len(elm_onehot), max(max_elms, 3),))
        ELM[:, :len(elm)] = elm_onehot[elm-1,:].T

        latt = crystal.lattice
        LATT = np.array((latt.abc, latt.angles))
        LATT = np.pad(LATT, ((0, 0), (0, max(max_elms, 3)-LATT.shape[1])), constant_values=0)
        
        # Obtain site coordinate matrix(Adding rotation transformations）
        SITE_COOR = np.array([site.frac_coords for site in crystal])
        rot_mat = calc_rot_mat(SITE_COOR)

        rotation_matrices.append(rot_mat.numpy())

        SITE_COOR = torch.tensor(SITE_COOR, dtype=torch.float)
        SITE_COOR = SITE_COOR @ rot_mat
        SITE_COOR = SITE_COOR.numpy()

        SITE_COOR = np.pad(SITE_COOR, ((0, max_sites-SITE_COOR.shape[0]), 
                                       (0, max(max_elms, 3)-SITE_COOR.shape[1])), constant_values=0)
        

        elm_inverse = np.zeros(len(crystal), dtype=int)
        for count, e in enumerate(elm):
            elm_inverse[np.argwhere(site_elm == e)] = count
        SITE_OCCU = OneHotEncoder().fit_transform(elm_inverse[:,np.newaxis]).toarray()
        SITE_OCCU = np.pad(SITE_OCCU, ((0, max_sites-SITE_OCCU.shape[0]),
                                       (0, max(max_elms, 3)-SITE_OCCU.shape[1])), constant_values=0)



        ELM_PROP = np.zeros((len(elm_prop[1]), max(max_elms, 3),))
        ELM_PROP[:, :len(elm)] = np.array([elm_prop[e] for e in elm]).T
        

        REAL = np.concatenate((ELM, LATT, SITE_COOR, SITE_OCCU, np.zeros((1, max(max_elms, 3))), ELM_PROP), axis=0)
        

        recip_latt = latt.reciprocal_lattice_crystallographic
        hkl, g_hkl, ind, _ = recip_latt.get_points_in_sphere([[0, 0, 0]], [0, 0, 0], 1.297, zip_results=False)
        if len(hkl) < 60:
            hkl, g_hkl, ind, _ = recip_latt.get_points_in_sphere([[0, 0, 0]], [0, 0, 0], 1.4, zip_results=False)
        not_zero = g_hkl!=0
        hkl = hkl[not_zero,:]
        g_hkl = g_hkl[not_zero]

        hkl = hkl.astype('int16')

        hkl_sum = np.sum(np.abs(hkl),axis=1)
        h = -hkl[:,0]
        k = -hkl[:,1]
        l = -hkl[:,2]
        hkl_idx = np.lexsort((l,k,h,hkl_sum))

        hkl_idx = hkl_idx[:59]
        hkl = hkl[hkl_idx,:]
        g_hkl = g_hkl[hkl_idx]

        k_dot_r = np.einsum('ij,kj->ik', hkl, SITE_COOR[:, :3])

        F_hkl = np.matmul(np.pad(ELM_PROP[:,elm_inverse], ((0, 0),
                                                           (0, max_sites-len(elm_inverse))), constant_values=0),
                          np.pi*k_dot_r.T)
        

        RECIP = np.zeros((REAL.shape[0], 59,))

        RECIP[-ELM_PROP.shape[0]-1, :] = g_hkl
        RECIP[-ELM_PROP.shape[0]:, :] = F_hkl


        RICR.append(np.concatenate([REAL, RECIP], axis=1))
        
        if return_Nsites:
            Nsites.append(len(crystal))
    RICR = np.stack(RICR)
    
    if not return_Nsites:
        return RICR
    else:
        return RICR, np.array(Nsites), np.array(rotation_matrices)