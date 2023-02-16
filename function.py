#%%
import torch
import numpy as np


def quaternion2rotation(quat):
    assert (quat.shape[1] == 4)
    # normalize first
    quat = quat / quat.norm(p=2, dim=1).view(-1, 1)

    a = quat[:, 0]
    b = quat[:, 1]
    c = quat[:, 2]
    d = quat[:, 3]

    a2 = a * a
    b2 = b * b
    c2 = c * c
    d2 = d * d
    ab = a * b
    ac = a * c
    ad = a * d
    bc = b * c
    bd = b * d
    cd = c * d

    # s = a2 + b2 + c2 + d2

    m0 = a2 + b2 - c2 - d2
    m1 = 2 * (bc - ad)
    m2 = 2 * (bd + ac)
    m3 = 2 * (bc + ad)
    m4 = a2 - b2 + c2 - d2
    m5 = 2 * (cd - ab)
    m6 = 2 * (bd - ac)
    m7 = 2 * (cd + ab)
    m8 = a2 - b2 - c2 + d2

    return torch.stack((m0, m1, m2, m3, m4, m5, m6, m7, m8), dim=1).view(-1, 3, 3)

def rotation2quaternion(M):
    tr = np.trace(M)
    m = M.reshape(-1)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[7] - m[5]) / s
        y = (m[2] - m[6]) / s
        z = (m[3] - m[1]) / s
    elif m[0] > m[4] and m[0] > m[8]:
        s = np.sqrt(1.0 + m[0] - m[4] - m[8]) * 2
        w = (m[7] - m[5]) / s
        x = 0.25 * s
        y = (m[1] + m[3]) / s
        z = (m[2] + m[6]) / s
    elif m[4] > m[8]:
        s = np.sqrt(1.0 + m[4] - m[0] - m[8]) * 2
        w = (m[2] - m[6]) / s
        x = (m[1] + m[3]) / s
        y = 0.25 * s
        z = (m[5] + m[7]) / s
    else:
        s = np.sqrt(1.0 + m[8] - m[0] - m[4]) * 2
        w = (m[3] - m[1]) / s
        x = (m[2] + m[6]) / s
        y = (m[5] + m[7]) / s
        z = 0.25 * s
    Q = np.array([w, x, y, z]).reshape(-1)
    return Q

def compute_loss(intrinsic, pt_3d, predQ, predT, gtQ, gtT):
    q1 = predQ
    t1 = predT
    q2 = gtQ
    t2 = gtT
    r1 = quaternion2rotation(q1)
    r2 = quaternion2rotation(q2)
    # 
    # compute error in 2D reprojection
    res1 = torch.bmm(r1, pt_3d.transpose(1, 2)) + t1.unsqueeze(dim=2)
    res2 = torch.bmm(r2, pt_3d.transpose(1, 2)) + t2.unsqueeze(dim=2)
    # 
    res1 = torch.bmm(intrinsic, res1)
    res2 = torch.bmm(intrinsic, res2)
    # 
    res1x = res1[:,0,:] / res1[:,2,:]
    res1y = res1[:,1,:] / res1[:,2,:]
    res2x = res2[:,0,:] / res2[:,2,:]
    res2y = res2[:,1,:] / res2[:,2,:]
    # 
    res1 = torch.cat((res1x.unsqueeze(-1), res1y.unsqueeze(-1)), dim=-1)
    res2 = torch.cat((res2x.unsqueeze(-1), res2y.unsqueeze(-1)), dim=-1)
    # 
    diff = (res1-res2).norm(dim=1).mean(dim=1)
    return diff.mean()



