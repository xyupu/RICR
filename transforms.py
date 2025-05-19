import numpy as np
import torch
import pdb


def calc_rot_mat(pos):
    if isinstance(pos, torch.Tensor):
        pos = pos.numpy()

    pos_centered = pos - pos.mean(0).reshape((1, -1))

    _, _, vt = np.linalg.svd(pos_centered)

    rot_mat = vt.T
    trans = pos_centered @ rot_mat

    ref_node = np.argmax(np.linalg.norm(trans, axis=1))
    ref_coord = trans[ref_node]

    mask = np.ones(pos.shape[1])
    mask[ref_coord < 0] = -1
    mask = mask.reshape((1, pos.shape[1]))

    rot_mat = rot_mat * mask


    return torch.from_numpy(rot_mat).float()


if __name__ == '__main__':
    pos = np.random.rand(10, 3)

    new_x = np.random.rand(3)
    new_x = new_x / np.linalg.norm(new_x)
    new_z = np.cross(new_x, np.random.rand(3))
    new_z = new_z / np.linalg.norm(new_z)
    new_y = np.cross(new_z, new_x)
    new_y = new_y / np.linalg.norm(new_y)
    rotation = np.linalg.inv(np.stack([new_x, new_y, new_z]))

    _pos = pos @ rotation + np.random.rand(3)

    invariant = get_invariant_pos(pos)
    _invariant = get_invariant_pos(_pos)

    assert np.allclose(invariant['trans'], _invariant['trans'])
    print('Passed.')