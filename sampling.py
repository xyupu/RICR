import joblib, os
import numpy as np
from tqdm import tqdm
from ase.io import write
from ase import spacegroup
from pymatgen.ext.matproj import MPRester

def get_info(RICR_designs,
             max_elms=3,
             max_sites=20,
             elm_str=joblib.load('data/element.pkl'),
             to_CIF=True,
             check_uniqueness=True,
             mp_api_key=None,
             ):


    Ntotal_elms = len(elm_str)
    pred_elm = np.argmax(RICR_designs[:, :Ntotal_elms, :max_elms], axis=1)

    def get_formula(RICR_designs, ):

        pred_for_array = np.zeros((RICR_designs.shape[0], max_sites))
        pred_formula = []
        pred_site_occu = RICR_designs[:, Ntotal_elms + 2 + max_sites:Ntotal_elms + 2 + 2 * max_sites, :max_elms]
        temp = np.repeat(np.expand_dims(np.max(pred_site_occu, axis=2), axis=2), max_elms, axis=2)
        pred_site_occu[pred_site_occu < temp] = 0
        pred_site_occu[pred_site_occu < 0.05] = 0
        pred_site_occu = np.ceil(pred_site_occu)
        for i in range(len(RICR_designs)):
            pred_for_array[i] = pred_site_occu[i].dot(pred_elm[i])

            if np.all(pred_for_array[i] == 0):
                pred_formula.append([elm_str[0]])
            else:
                temp = pred_for_array[i]
                temp = temp[:np.where(temp > 0)[0][-1] + 1]
                temp = temp.tolist()
                pred_formula.append([elm_str[int(j)] for j in temp])
        return pred_formula

    pred_formula = get_formula(RICR_designs)
    pred_abc = RICR_designs[:, Ntotal_elms, :3]
    pred_ang = RICR_designs[:, Ntotal_elms + 1, :3]
    pred_latt = np.concatenate((pred_abc, pred_ang), axis=1)
    pred_site_coor = []
    pred_site_coor_ = RICR_designs[:, Ntotal_elms + 2:Ntotal_elms + 2 + max_sites, :3]
    for i, c in enumerate(pred_formula):
        Nsites = len(c)
        pred_site_coor.append(pred_site_coor_[i, :Nsites, :])

    if check_uniqueness:
        assert mp_api_key != None, "You need a mp_api_key to check the uniqueness of designed CIFs!"
        mpr = MPRester(mp_api_key)
        ind = []
        op = tqdm(range(len(pred_formula)))
        for i in op:
            op.set_description("Checking uniqueness of designed compostions in the Materials Project database")
            formula = ''.join(pred_formula[i])
            query = mpr.summary.search(formula=formula)
            if not query:
                ind.append(i)
    else:
        ind = list(np.arange(len(pred_formula)))

    if to_CIF:
        os.makedirs('designed_CIFs', exist_ok=True)

        op = tqdm(ind)
        for i, j in enumerate(op):
            op.set_description("Writing designed crystals as CIFs")

            try:
                crystal = spacegroup.crystal(pred_formula[j],
                                             basis=pred_site_coor[j],
                                             cellpar=pred_latt[j])
                write('designed_CIFs_3/' + str(i) + '.cif', crystal)
            except:
                pass

    if check_uniqueness:
        ind_unique = ind
        return pred_formula, pred_abc, pred_ang, pred_latt, pred_site_coor, ind_unique
    else:
        return pred_formula, pred_abc, pred_ang, pred_latt, pred_site_coor



RICR_designs = np.load('your_ricr.npy')
pred_formula, pred_abc, pred_ang, pred_latt, pred_site_coor, ind_unique = get_info(RICR_designs,
                                                                                   max_elms=4,
                                                                                   max_sites=40,
                                                                                   elm_str=joblib.load('data/element.pkl'),
                                                                                   to_CIF=True,
                                                                                   check_uniqueness=True,
                                                                                   mp_api_key='key',
                                                                                   )