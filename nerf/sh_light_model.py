import os
import cv2
import math
import glob
import torch
import imageio
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch import Tensor
import tinycudann as tcnn
import torch.nn.functional as F
from tools.shape_tools import write_ply_rgb


zero_band_coeffs_list = [0.8754318, 1.023545, 0.449686, 0.]

####################################################################################################################
# Code from svox2


SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
SH_C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]

def svox2_eval_sh_bases(basis_dim : int, dirs : torch.Tensor):
    """
    Evaluate spherical harmonics bases at unit directions,
    without taking linear combination.
    At each point, the final result may the be
    obtained through simple multiplication.
    :param basis_dim: int SH basis dim. Currently, 1-25 square numbers supported
    :param dirs: torch.Tensor (..., 3) unit directions
    :return: torch.Tensor (..., basis_dim)
    """
    result = torch.empty((*dirs.shape[:-1], basis_dim), dtype=dirs.dtype, device=dirs.device)
    result[..., 0] = SH_C0
    if basis_dim > 1:
        x, y, z = dirs.unbind(-1)
        result[..., 1] = -SH_C1 * y;
        result[..., 2] = SH_C1 * z;
        result[..., 3] = -SH_C1 * x;
        if basis_dim > 4:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result[..., 4] = SH_C2[0] * xy;
            result[..., 5] = SH_C2[1] * yz;
            result[..., 6] = SH_C2[2] * (2.0 * zz - xx - yy);
            result[..., 7] = SH_C2[3] * xz;
            result[..., 8] = SH_C2[4] * (xx - yy);

            if basis_dim > 9:
                result[..., 9] = SH_C3[0] * y * (3 * xx - yy);
                result[..., 10] = SH_C3[1] * xy * z;
                result[..., 11] = SH_C3[2] * y * (4 * zz - xx - yy);
                result[..., 12] = SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy);
                result[..., 13] = SH_C3[4] * x * (4 * zz - xx - yy);
                result[..., 14] = SH_C3[5] * z * (xx - yy);
                result[..., 15] = SH_C3[6] * x * (xx - 3 * yy);

                if basis_dim > 16:
                    result[..., 16] = SH_C4[0] * xy * (xx - yy);
                    result[..., 17] = SH_C4[1] * yz * (3 * xx - yy);
                    result[..., 18] = SH_C4[2] * xy * (7 * zz - 1);
                    result[..., 19] = SH_C4[3] * yz * (7 * zz - 3);
                    result[..., 20] = SH_C4[4] * (zz * (35 * zz - 30) + 3);
                    result[..., 21] = SH_C4[5] * xz * (7 * zz - 3);
                    result[..., 22] = SH_C4[6] * (xx - yy) * (7 * zz - 1);
                    result[..., 23] = SH_C4[7] * xz * (xx - 3 * yy);
                    result[..., 24] = SH_C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy));
    return result


def svox2_sh_eval(degree, sh_param, dirs):
    sh_bases = svox2_eval_sh_bases((degree+1)**2, dirs)
    if sh_param.dim == 2:
        sh_param = sh_param[:, None]  # SH, 3
    result = (sh_param[:, :(degree+1)**2] * sh_bases[..., None]).sum(dim=1)
    return result

####################################################################################################################

### -- Functions for SH rotation

# ---- The following functions are used to implement SH rotation computations
#      based on the recursive approach described in [1, 4]. The names of the
#      functions correspond with the notation used in [1, 4].
# See http://en.wikipedia.org/wiki/Kronecker_delta
nearbymargin = lambda x, y: abs(x-y) < 1e-8

kronecker_delta = lambda i, j: 1.0 if i == j else 0.0

def centered_elem(r, i, j):
    '''
        // [4] uses an odd convention of referring to the rows and columns using
        // centered indices, so the middle row and column are (0, 0) and the upper
        // left would have negative coordinates.
        //
        // This is a convenience function to allow us to access an Eigen::MatrixXd
        // in the same manner, assuming r is a (2l+1)x(2l+1) matrix.
    '''
    offset = (r.shape[1] - 1) // 2
    return r[:, i + offset, j + offset]
    
def P(i, a, b, l, r):
    '''
        // P is a helper function defined in [4] that is used by the functions U, V, W.
        // This should not be called on its own, as U, V, and W (and their coefficients)
        // select the appropriate matrix elements to access (arguments @a and @b).
    '''
    if b == l:
        return centered_elem(r[1], i, 1) * centered_elem(r[l - 1], a, l - 1) - centered_elem(r[1], i, -1) * centered_elem(r[l - 1], a, -l + 1)
    elif b == -l:
        return centered_elem(r[1], i, 1) * centered_elem(r[l - 1], a, -l + 1) + centered_elem(r[1], i, -1) * centered_elem(r[l - 1], a, l - 1)
    else:
        return centered_elem(r[1], i, 0) * centered_elem(r[l - 1], a, b)

# The functions U, V, and W should only be called if the correspondingly
# named coefficient u, v, w from the function ComputeUVWCoeff() is non-zero.
# When the coefficient is 0, these would attempt to access matrix elements that
# are out of bounds. The list of rotations, @r, must have the @l - 1
# previously completed band rotations. These functions are valid for l >= 2.

def U(m, n, l, r):
    '''
        // Although [1, 4] split U into three cases for m == 0, m < 0, m > 0
        // the actual values are the same for all three cases
    '''
    return P(0, m, n, l, r)

def V(m, n, l, r):
    if m == 0:
        return P(1, 1, n, l, r) + P(-1, -1, n, l, r)
    elif m > 0:
        return P(1, m - 1, n, l, r) * math.sqrt(1 + kronecker_delta(m, 1)) - P(-1, -m + 1, n, l, r) * (1 - kronecker_delta(m, 1))
    else:
        '''
            // Note there is apparent errata in [1,4,4b] dealing with this particular
            // case. [4b] writes it should be P*(1-d)+P*(1-d)^0.5
            // [1] writes it as P*(1+d)+P*(1-d)^0.5, but going through the math by hand,
            // you must have it as P*(1-d)+P*(1+d)^0.5 to form a 2^.5 term, which
            // parallels the case where m > 0.
        '''
        return P(1, m + 1, n, l, r) * (1 - kronecker_delta(m, -1)) + P(-1, -m - 1, n, l, r) * math.sqrt(1 + kronecker_delta(m, -1))

def W(m, n, l, r):
    if m == 0:
        # whenever this happens, w is also 0 so W can be anything
        return 0.0
    elif m > 0:
        return P(1, m + 1, n, l, r) + P(-1, -m - 1, n, l, r)
    else:
        return P(1, m - 1, n, l, r) - P(-1, -m + 1, n, l, r)

def compute_uvw(m, n, l):
    '''
        // Calculate the coefficients applied to the U, V, and W functions. Because
        // their equations share many common terms they are computed simultaneously.
    '''
    d = kronecker_delta(m, 0)
    denom = (2.0 * l * (2.0 * l - 1) if abs(n) == l else (l + n) * (l - n))

    u = math.sqrt((l + m) * (l - m) / denom)
    v = 0.5 * math.sqrt((1 + d) * (l + abs(m) - 1.0) * (l + abs(m)) / denom) * (1 - 2 * d)
    w = -0.5 * math.sqrt((l - abs(m) - 1) * (l - abs(m)) / denom) * (1 - d)
    return u,v,w

def compute_band_rotation(l, rotations: list):
    '''
        // Calculate the (2l+1)x(2l+1) rotation matrix for the band @l.
        // This uses the matrices computed for band 1 and band l-1 to compute the
        // matrix for band l. @rotations must contain the previously computed l-1
        // rotation matrices, and the new matrix for band l will be appended to it.
        //
        // This implementation comes from p. 5 (6346), Table 1 and 2 in [4] taking
        // into account the corrections from [4b].
        
        // The band's rotation matrix has rows and columns equal to the number of
        // coefficients within that band (-l <= m <= l implies 2l + 1 coefficients).
    '''
    rotation = torch.zeros(rotations[0].shape[0], 2*l+1, 2*l+1).float().type_as(rotations[0])
    for m in range(-l, l+1):
        for n in range(-l, l+1):
            u, v, w = compute_uvw(m, n, l)
            if not nearbymargin(u, 0):
                u *= U(m, n, l, rotations)
            if not nearbymargin(v, 0):
                v *= V(m, n, l, rotations)
            if not nearbymargin(w, 0):
                w *= W(m, n, l, rotations)

            rotation[:, m+l, n+l] = u + v + w
    rotations.append(rotation)

# For efficiency, the cosine lobe for normal = (0, 0, 1) as the first 9
# spherical harmonic coefficients are hardcoded below. This was computed by
# evaluating:
#   ProjectFunction(kIrradianceOrder, [] (double phi, double theta) {
#     return Clamp(Eigen::Vector3d::UnitZ().dot(ToVector(phi, theta)), 
#                  0.0, 1.0);
#   }, 10000000);
def create_rot_by_mat(order, R_mat):
    ret = []

    # l = 0
    ret.append(torch.ones([R_mat.shape[0], 1,1]).type_as(R_mat))

    # l = 1
    r = torch.stack([R_mat[:, 1, 1], -R_mat[:, 1, 2], R_mat[:, 1, 0], 
                    -R_mat[:, 2, 1], R_mat[:, 2, 2], -R_mat[:, 2, 0], 
                    R_mat[:, 0, 1], -R_mat[:, 0, 2], R_mat[:, 0, 0]], dim=-1).reshape(-1, 3, 3)
    ret.append(r)

    for l in range(2, order+1):
        compute_band_rotation(l, ret)
    
    return ret

def rotate_coeff_by_rotmat(order, R_mat, coeff):
    '''
        input:
            order: int
            R_mat: tensor of shape [B1, 3, 3]
            coeff: tensor of shape [B2, N]
        return:
            tensor of shape [B1*B2, N]
    '''
    R_list = create_rot_by_mat(order, R_mat)

    # transform one band(order) at a time
    # equivalent to a huge matrix multipilication, 
    batch_rot = R_mat.shape[0]
    batch_coeff = coeff.shape[0]
    batch = batch_rot * batch_coeff
    ret = coeff[None,:,:].repeat([batch_rot, 1, 1]).reshape(batch, -1)
    for l in range(order+1):
        band_coeff = coeff[None, :, l*l:(l+1)*(l+1), None].repeat([batch_rot, 1, 1, 1]).reshape(batch, 2*l+1, 1) # B x 2l+1 x 1
        rot_coeff = R_list[l][:, None, :, :].repeat(1, batch_coeff, 1, 1).reshape(batch, 2*l+1, 2*l+1) # B x 2l+1 x 2l+1
        band_coeff = torch.bmm(rot_coeff, band_coeff) # B x 2l+1 x 1
        ret[:, l*l:(l+1)*(l+1)] = band_coeff[:, :, 0]
    
    return ret

def rotate_coeff_by_normal(order, normal, coeff):
    normal = normal / (normal.norm(dim=1, keepdim=True)+1e-9)
    z = torch.zeros_like(normal)
    z[:, 2] = 1 
    x = torch.cross(normal, z)
    x = x / (x.norm(dim=1, keepdim=True)+1e-9)
    y = torch.cross(normal, x)
    R_mat = torch.stack([x, y, normal], dim=-1)
    I_mat = torch.from_numpy(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])).float().type_as(R_mat)
    I_mat_2 = torch.from_numpy(np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])).float().type_as(R_mat)
    R_mat[normal[:,2]>0.999] = I_mat
    R_mat[normal[:,2]<-0.999] = I_mat_2            
    return rotate_coeff_by_rotmat(order, R_mat, coeff)

def rotate_coeffs(orders, R_mat, coeffs):    
    rotated_coeffs = rotate_coeff_by_rotmat(orders, R_mat, coeffs.transpose(1, 0))
    rotated_coeffs = rotated_coeffs.reshape(-1, 3, get_coef_count(orders)).transpose(1, 2)
    return rotated_coeffs

####################################################################################################################


get_index = lambda l, m: l*(l+1)+m
get_coef_count = lambda order: (order+1)*(order+1)

# factorial: x!
factorial_cache = [1, 1, 2, 6, 24, 120, 720, 5040,
                    40320, 362880, 3628800, 39916800,
                    479001600, 6227020800,
                    87178291200, 1307674368000]
factorial = lambda x: factorial_cache[x] if x <= 15 else math.factorial(x)


# double factorial: x!!
dbl_factorial_cache = [1, 1, 2, 3, 8, 15, 48, 105,
                        384, 945, 3840, 10395, 46080,
                        135135, 645120, 2027025]
double_factorial = lambda x: dbl_factorial_cache[x] if x <= 15 else torch.prod(torch.arange(x, 0, -2))


def normalize(tensor):
    return tensor / (tensor.norm(dim=-1, keepdim=True)+1e-9)


def isclose(x, val, threshold = 1e-6):
    return torch.abs(x - val) <= threshold


def safe_sqrt(x):
    sqrt_in = torch.relu(torch.where(isclose(x, 0.0), torch.ones_like(x) * 1e-6, x))
    return torch.sqrt(sqrt_in)


def safe_pow(x, p):
    sqrt_in = torch.relu(torch.where(isclose(x, 0.0), torch.ones_like(x) * 1e-6, x))
    return torch.pow(sqrt_in, p)


# Generate matrics for fast calculation
def pre_calc_sh_mat():
    ar = torch.arange(4)
    mat_x, mat_y, mat_z = torch.meshgrid(ar, ar, ar) # [4,4,4]
    mat = torch.stack([mat_x, mat_y, mat_z], dim=-1) # [4,4,4,3]

    # Assigning constants of the real SH polynomials
    s = torch.zeros([16, 4, 4, 4])
    # Level 0
    s[0][0][0][0] = 0.282095    # 0.282095

    # Level 1
    s[1][0][1][0] = -0.488603   # -0.488603 * y
    s[2][0][0][1] = 0.488603   # 0.488603 * z 
    s[3][1][0][0] = -0.488603   # -0.488603 * x 

    # Level 2
    s[4][1][1][0] = 1.092548   # 1.092548 * x * y
    s[5][0][1][1] = -1.092548; # -1.092548 * y * z
    s[6][2][0][0] = -0.315392; s[6][0][2][0] = -0.315392; s[6][0][0][2] = 0.315392 * 2.0   # 0.315392 * (- x * x - y * y + 2.0 * z * z)
    s[7][1][0][1] = -1.092548   # -1.092548 * x * z
    s[8][2][0][0] = 0.546274; s[8][0][2][0] = -0.546274  # 0.546274 * (x * x - y * y)

    # Level 3
    s[9][2][1][0] = -0.590044 * 3.0; s[9][0][3][0] = 0.590044 # -0.590044 * y * (3.0 * x * x - y * y)
    s[10][1][1][1] = 2.890611    # 2.890611 * x * y * z
    s[11][0][1][2] = -0.457046 * 4.0; s[11][2][1][0] = 0.457046; s[11][0][3][0] = 0.457046 # -0.457046 * y * (4.0 * z * z - x * x - y * y)
    s[12][0][0][3] = 0.373176 * 2.0; s[12][2][0][1] = -0.373176 * 3.0; s[12][0][2][1] = -0.373176 * 3.0 # 0.373176 * z * (2.0 * z * z - 3.0 * x * x - 3.0 * y * y)
    s[13][1][0][2] = -0.457046 * 4.0; s[13][3][0][0] = 0.457046; s[13][1][2][0] = 0.457046 # -0.457046 * x * (4.0 * z * z - x * x - y * y)
    s[14][2][0][1] = 1.445306; s[14][0][2][1] = -1.445306 # 1.445306 * z * (x * x - y * y)
    s[15][3][0][0] = -0.590044; s[15][1][2][0] = 0.590044 * 3.0 # -0.590044 * x * (x * x - 3.0 * y * y)

    # select terms that are used in at least one polynomial
    valid = torch.nonzero((torch.max(s**2, dim=0)[0]>0))
    idx = torch.zeros_like(s[0]).long() # 4 x 4 x 4
    idx[valid[:,0], valid[:,1], valid[:,2]] = torch.arange(len(valid), device=valid.device) # 4 x 4 x 4
    sh_s = s[:, valid[:,0], valid[:,1], valid[:,2]] # 16 x N_valid
    sh_power_num = valid # N_valid, 3
    return sh_power_num, sh_s


def angle2xyz(angle: torch.FloatTensor):
    # Y up
    # angle: [phi, theta], tensor shape: n, 2
    r = torch.sin(angle[..., 1])
    return torch.stack([r * torch.cos(angle[..., 0]), torch.cos(angle[..., 1]), r * torch.sin(angle[..., 0])], -1).type_as(angle)


def xyz2angle(xyz: torch.FloatTensor):
    # Y up
    return torch.stack([torch.atan2(xyz[:, 2], xyz[:, 0]), torch.acos(torch.clamp(xyz[:, 1], -1.0, 1.0))], dim=-1).type_as(xyz)


def _eval_legendre(l, m, x):

    '''
        // Evaluate the associated Legendre polynomial of degree @l and order @m at
        // coordinate @x. The inputs must satisfy:
        // 1. l >= 0
        // 2. 0 <= m <= l
        // 3. -1 <= x <= 1
        // See http://en.wikipedia.org/wiki/Associated_Legendre_polynomials
        //
        // This implementation is based off the approach described in [1],
        // instead of computing Pml(x) directly, Pmm(x) is computed. Pmm can be
        // lifted to Pmm+1 recursively until Pml is found
    '''

    # Compute Pmm(x) = (-1)^m(2m - 1)!!(1 - x^2)^(m/2), where !! is the double factorial.
    pmm = torch.ones_like(x)

    if m > 0:
        sign = (1 if m % 2 == 0 else -1)
        pmm = sign * double_factorial(2 * m - 1) * torch.pow(1 - x * x, m / 2.0)

    if l == m:
        # Pml is the same as Pmm so there's no lifting to higher bands needed
        return pmm

    # Compute Pmm+1(x) = x(2m + 1)Pmm(x)
    pmm1 = x * (2 * m + 1) * pmm
    if l == m + 1:
        # Pml is the same as Pmm+1 so we are done as well
        return pmm1

    # Use the last two computed bands to lift up to the next band until l is
    # reached, using the recurrence relationship:
    # Pml(x) = (x(2l - 1)Pml-1 - (l + m - 1)Pml-2) / (l - m)
    for n in range(m+2, l+1):
        pmn = (x * (2 * n - 1) * pmm1 - (n + m - 1) * pmm) / (n - m)
        pmm = pmm1
        pmm1 = pmn

    return pmm1


def _eval_sh(l, m, angle=None, xyz=None):
    def _xyz2angle_z(xyz: torch.FloatTensor):
        # Z up
        return torch.stack([torch.atan2(xyz[:, 1], xyz[:, 0]), torch.acos(torch.clamp(xyz[:, 2], -1.0, 1.0))], dim=-1).type_as(xyz)
    def _angle_y2z(angle: torch.FloatTensor):
        return _xyz2angle_z(angle2xyz(angle))

    assert(l >= 0)
    assert(-l <= m and m <= l)
    assert((angle is None) != (xyz is None)) # only one of them should be filled

    # the original sh function uses z-axis as the top direction, while our method uses y-axis.
    # thus a convention of axes is required here.
    if angle is None:
        angle = _xyz2angle_z(xyz) 
    else:
        angle = _angle_y2z(angle)

    phi = angle[:, 0]
    theta = angle[:, 1]

    kml = math.sqrt((2.0 * l + 1) * factorial(l - abs(m)) / (4.0 * math.pi * factorial(l + abs(m))))

    if m > 0:
        return math.sqrt(2.0) * kml * torch.cos(m * phi) * _eval_legendre(l, m, torch.cos(theta))
    elif m < 0:
        return math.sqrt(2.0) * kml * torch.sin(-m * phi) * _eval_legendre(l, -m, torch.cos(theta))
    else: # m=0
        return kml * _eval_legendre(l, 0, torch.cos(theta))


def _eval_sh_sum(order, coeffs, angle=None, xyz=None):
    '''
        coeffs: [B, lm, 3]
        angle: [B, 3]
        xyz: [B, 3]
    '''
    if coeffs.ndim == 2:
        coeffs = coeffs[None,:,:]
    B = (angle.shape[0] if angle is not None else xyz.shape[0])
    sum = torch.zeros([B, coeffs.shape[-1]]).float().type_as(coeffs) # B, 3
    for l in range(order+1):
        for m in range(-l, l+1):
            sh = _eval_sh(l, m, angle, xyz) # B
            sum += sh[:, None] * coeffs[:,get_index(l, m)]
    return sum


def fast_sh_sum(order, power_num, s, coeffs, angle=None, xyz=None):
    '''
        power_num: [N_valid x 3], pre_computed buffer
        s: [lm x N_valid], pre_computed buffer
        
        coeffs: [B x lm x 3], input SH coefficients
        xyz: [B x 3], input coordinate
    '''
    assert(order <= 3)    
    
    if xyz is None:
        xyz = angle2xyz(angle)

    if len(xyz.shape) == 1:
        xyz = xyz[None,:]
    if coeffs.ndim == 2:
        coeffs = coeffs[None,:,:]

    coeff_count = get_coef_count(order)

    xyz_term = torch.pow(xyz[:,None,:], power_num[None,:,:]).prod(dim=-1) # B x N_Valid
    sh_val = torch.matmul(xyz_term, s.T)[:, :coeff_count] # B x lm

    return (sh_val[:,:,None] * coeffs).sum(dim=1) # B x 3


def render_irrandiance_sh_sum(coeffs, normal):
    '''Calculate irradiance using method proposed by Ramamoorthi et al., 2001'''
    if coeffs.ndim == 2:
        coeffs = coeffs[None]
    cosine_lobe = torch.FloatTensor([3.14, 2.09, 2.09, 2.09, 0.79, 0.79, 0.79, 0.79, 0.79]).type_as(normal)[None,:] / math.pi # 1 x 9
    # cosine_lobe = torch.FloatTensor([0.8754318, 1.023545, 1.023545, 1.023545, 0.449686, 0.449686, 0.449686, 0.449686, 0.449686]).type_as(normal)[None, :]
    coeffs = coeffs[:, :9, :] * cosine_lobe[:, :, None]
    # Bug in fast_sh_sum
    return svox2_sh_eval(2, coeffs, dirs=normal)  # _eval_sh_sum(2, coeffs, xyz=normal) #if not fast else fast_sh_sum(2, pow_num, s, coeffs, xyz=normal)


class SH_EnvmapMaterialNet(nn.Module):
    def __init__(self,
                 input_dim,
                 sh_order=3,
                 white_light=False,
                 use_specular=True):
        super().__init__()

        ############## SH Envmap ############
        self.sh_order = sh_order
        self.white_light = white_light
        color_dim = 3 if not white_light else 1
        init_light = torch.zeros([(self.sh_order+1)*(self.sh_order+1), color_dim])
        # initialize lighting with a white ambient light
        init_light[0, :] = 3
        self.envSHs = nn.Parameter(init_light, requires_grad=True)
        # pre-process the coefficients for fast SH intergration
        pow_num, s = pre_calc_sh_mat()
        self.sh_pow_num = nn.Parameter(pow_num, requires_grad=False)
        self.sh_s = nn.Parameter(s, requires_grad=False)
        self.gamma = 2.4
        self.min_glossiness = 1.

        ############## BRDF Network ############
        self.brdf_layer = tcnn.Network(
            n_input_dims=input_dim,
            n_output_dims=5,  # albedo[3], specular[1], glossiness[1]
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "Relu",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 3 - 1,
            },
        )
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.import_envmap = False
        self.envSHs_import = None
        self.envSHs_import_vis = None
        self.sh_order_import = None
        self.sh_order_import_vis = None
        self.use_specular = use_specular
        self.name = 'sh'
    
    def forward(self, geo_feat, normals_primary, view_dirs, **kwargs):
        shade_visibility = kwargs['shade_visibility'] if 'shade_visibility' in kwargs.keys() else False
        if shade_visibility:
            normals_secondary = kwargs['normal_secondary']
        else:
            normals_secondary = None
        
        if self.import_envmap and not self.training:
            if not shade_visibility:
                envSHs = self.envSHs_import[None]
                sh_order = self.sh_order_import
            else:
                prob_idx = (normals_secondary[:, None] * self.probes[None]).sum(dim=-1).argmax(dim=-1)
                envSHs = self.envSHs_import_vis[prob_idx]
                sh_order = self.sh_order_import_vis
        else:
            envSHs = self.envSHs[None]
            sh_order = self.sh_order

        prefix = geo_feat.shape[:-1]
        geo_feat = geo_feat.reshape([-1, geo_feat.shape[-1]])
        normals_primary = normals_primary.reshape([-1, normals_primary.shape[-1]])
        view_dirs = view_dirs.reshape([-1, view_dirs.shape[-1]])

        brdf = self.brdf_layer(geo_feat)
        albedo = self.sigmoid(brdf[..., :3])
        specular = self.sigmoid(brdf[..., 3:4])
        glossiness = self.softplus(brdf[...,4:5]) + self.min_glossiness

        # Diffusion Color: from the first two levels of SH only
        diffuse_rgb = render_irrandiance_sh_sum(envSHs[:, :9,:3], normals_primary)
        diffuse_rgb = diffuse_rgb.reshape([*albedo.shape[:-1], -1]).clamp(0)  # N_rays*N_samples, 3
        diffuse = albedo * diffuse_rgb  # N_rays*N_samples, 3

        if self.use_specular:
            # reflection formula: w_i = 2 * |w_o * n| * n - w_o
            rays_d = normalize(view_dirs)  # N_rays*N_samples, 3
            cos_theta = -(rays_d * normals_primary).sum(dim=-1, keepdim=True)  # N_rays*N_samples, 1
            reflect_d = normalize(2 * cos_theta * normals_primary + rays_d)  # N_rays*N_samples, 3

            order_coeff = torch.arange(0, envSHs.shape[0], device=envSHs.device)[:, None]  # 1, N_sh, 1
            order_coeff = torch.pow(order_coeff, 0.5).floor()
            s = glossiness  # N_rays*N_samples, 1
            sh_coeff = torch.exp(-order_coeff * order_coeff / 2 / s)[..., None] * envSHs[:, :9, :3] # 1, N_sh, 3

            specular_rgb = render_irrandiance_sh_sum(sh_coeff, reflect_d)
            specular = (specular * specular_rgb).reshape([*albedo.shape[:-1], -1])
        else:
            specular = torch.zeros_like(diffuse)
        
        color = diffuse + specular
        color = color.clamp(0).reshape([*prefix, -1])
        diffuse = diffuse.clamp(0, 1).reshape([*prefix, -1])
        specular = specular.clamp(0, 1).reshape([*prefix, -1])
        if specular.shape[-1] == 1:
            specular = specular.expand_as(color)
        albedo = albedo.clamp(0, 1).reshape([*prefix, -1])
        # Tone Mapping
        gamma = self.gamma if 'gamma' not in kwargs.keys() or kwargs['gamma'] is None else kwargs['gamma']
        color = safe_pow(color, 1 / gamma)
        diffuse = safe_pow(diffuse,1 / gamma)
        specular = safe_pow(specular, 1 / gamma)
        return color, specular, diffuse, albedo

    def save_envmap(self, sv_path, H=256, W=512):
        sv_path = self.specific_path(sv_path)
        envmap, viewdirs = SH2Envmap(self.envSHs, H=H, W=W, upper_hemi=False, clamp=False)
        envmap = envmap.detach().cpu().numpy()
        viewdirs = viewdirs.detach().cpu().numpy()
        save_envmap(envshs=self.envSHs.detach().cpu().numpy(), envmap=envmap, viewdirs=viewdirs, path=sv_path)
    
    def load_envmap(self, path, log_path, force_white=True):
        if os.path.exists(self.specific_path(path) + '.npy'):
            print('Loading envmap from ', self.specific_path(path) + '.npy')
            self.envSHs_import = nn.Parameter(torch.from_numpy(np.load(self.specific_path(path) + '.npy')).cuda(), requires_grad=False)
        else:
            files = glob.glob(path + '.*')
            files = [file for file in files if file.endswith('.jpeg') or file.endswith('.png') or file.endswith('.jpg') or file.endswith('JPEG')]
            if len(files) == 0:
                print('No envmap found: ', path)
                return False
            file = files[0]
            print('Optimizing SHs towards ', file, ' ...')
            envmap = image2envmap(file, force_white=force_white)
            envSHs = EnvMap2SH(envmap, sh_order=self.sh_order, log_path=log_path, min_loss=1e-5, sv_path=self.specific_path(path), fast=True, pow_num=self.sh_pow_num, s=self.sh_s)
            self.envSHs_import = nn.Parameter(envSHs, requires_grad=False)
        self.sh_order_import = int(self.envSHs_import.shape[0] ** .5) - 1
        self.load_envmap_with_visibility(path, log_path=log_path, force_white=force_white)
        torch.cuda.empty_cache()
        print('Load Envmap Done!')
        self.import_envmap = True
        return True
    
    def load_envmap_with_visibility(self, path, log_path, force_white=True):
        if os.path.exists(self.specific_path(path) + '_vis.npz'):
            print('Loading envmap from ', self.specific_path(path) + '_vis.npz')
            data = np.load(self.specific_path(path) + '_vis.npz')
            self.envSHs_import_vis = nn.Parameter(torch.from_numpy(data['envSHs']).cuda(), requires_grad=False)
            self.probes = nn.Parameter(torch.from_numpy(data['probes']).cuda(), requires_grad=False)
        else:
            print('Loading envmaps with visibility... It may take a while for the first time...')
            if self.envSHs_import is None:
                self.load_envmap(path=path, log_path=log_path, force_white=force_white)
            H, W = 8, 8
            phi, theta = torch.meshgrid([torch.linspace(np.pi / H, np.pi, H), torch.linspace(-0.5*np.pi, 1.5*np.pi, W)])
            self.probes = torch.stack([torch.cos(theta) * torch.sin(phi), torch.cos(phi), torch.sin(theta) * torch.sin(phi)],
                                dim=-1).cuda().reshape([-1, 3])
            visibility_lobe = torch.FloatTensor([0.8754318, 0, 1.023545, 0, 0., 0., 0.449686, 0., 0.]).type_as(self.probes)[None, :]
            visibility = rotate_coeff_by_normal(2, self.probes, visibility_lobe)[..., None].repeat([1, 1, 3])
            self.envSHs_import_vis = torch.zeros([self.probes.shape[0], 9, 3]).type_as(self.envSHs_import)
            for i in tqdm(range(self.envSHs_import_vis.shape[0])):
                self.envSHs_import_vis[i] = fit_product_of_SHs(self.envSHs_import[None, :9], visibility[i:i+1])
            np.savez(self.specific_path(path) + '_vis', envSHs=self.envSHs_import_vis.detach().cpu().numpy(), probes=self.probes.cpu().detach().numpy())
        self.sh_order_import_vis = int(self.envSHs_import_vis.shape[0] ** .5) - 1
        torch.cuda.empty_cache()
        print('Load Envmap with Visibility Done!')
        return True        
    
    def specific_path(self, path):
        return path + '_' + self.name
    
    def switch_envmap_import(self):
        if self.envSHs_import is None:
            print('No Imported Light Envmap')
            self.import_envmap = False
        else:
            self.import_envmap = not self.import_envmap
        return self.import_envmap
    
    def switch_envmap_visbility(self):
        if self.envSHs_import is None:
            print('No Imported Light Envmap')
            self.import_envmap = False
        else:
            self.with_visibility = not self.with_visibility
        return self.import_envmap


def fit_product_of_SHs(sh1, sh2):
    from torch.nn import Parameter
    from torch.optim import Adam
    from nerf.spherical_harmonics import evaluate_spherical_harmonics
    sh = Parameter(sh1.clone())
    optimizer = Adam([sh], lr=1e-3)
    batch_size = 1024
    rounds = 800
    for _ in range(rounds):
        phi = 2 * np.pi * torch.rand([batch_size], device=sh.device)
        theta = torch.arccos(1 - 2 * torch.rand([batch_size], device=sh.device))
        points = torch.stack([torch.cos(phi) * torch.sin(theta), torch.sin(phi) * torch.sin(theta), torch.cos(theta)], dim=-1)
        gt = evaluate_spherical_harmonics(2, sh1.permute(0, 2, 1), points) * evaluate_spherical_harmonics(2, sh2.permute(0, 2, 1), points)
        pred = evaluate_spherical_harmonics(2, sh.permute(0, 2, 1), points)
        loss = ((gt - pred) ** 2).sum(dim=-1).mean()
        loss.backward()
        optimizer.step()
    return sh


def SH2Envmap(envSHs, H, W, upper_hemi=False, clamp=False):
    # exactly same convetion as Mitsuba, check envmap_convention.png
    if upper_hemi:
        phi, theta = torch.meshgrid([torch.linspace(0., np.pi/2., H), torch.linspace(-0.5*np.pi, 1.5*np.pi, W)])
    else:
        phi, theta = torch.meshgrid([torch.linspace(0., np.pi, H), torch.linspace(-0.5*np.pi, 1.5*np.pi, W)])
    viewdirs = torch.stack([torch.cos(theta) * torch.sin(phi), torch.cos(phi), torch.sin(theta) * torch.sin(phi)],
                           dim=-1)    # [H, W, 3]
    viewdirs = viewdirs.to(envSHs.device)
    viewdirs = viewdirs.reshape([-1, 3])
    rgb = svox2_sh_eval(2, envSHs[None, :9,:3], dirs=viewdirs)
    if clamp:
        rgb = rgb.clamp(0, 1)
    envmap = rgb.reshape((H, W, rgb.shape[-1]))
    viewdirs = viewdirs.reshape([*envmap.shape[:-1], -1])
    return envmap, viewdirs


def EnvMap2SH(envmap, sh_order=3, log_path='./logs/', min_loss=1e-5, sv_path='./logs/sh', fast=False, pow_num=None, s=None):
    print('Fitting Environment Map with Spherical Harmonics ...')
    # ground-truth envmap
    gt_envmap = torch.from_numpy(envmap).cuda()
    H, W = gt_envmap.shape[:2]
    os.makedirs(log_path, exist_ok=True)
    assert (os.path.isdir(log_path))
    
    init_light = torch.zeros([(sh_order+1)*(sh_order+1), 3]).cuda()
    envSHs = nn.Parameter(init_light, requires_grad=True)

    optimizer = torch.optim.Adam([envSHs], lr=1e-2)
    N_iter = 5000

    for step in tqdm(range(N_iter)):
        optimizer.zero_grad()
        env_map, viewdirs = SH2Envmap(envSHs, H=H, W=W, upper_hemi=False, clamp=False)
        loss = torch.mean((env_map - gt_envmap) * (env_map - gt_envmap))
        loss.backward()
        optimizer.step()
        if step % 30 == 0:
            print('step: {}, loss: {}'.format(step, loss.item()))
        if step % 1000 == 0:
            envmap_check = env_map.clone().detach().cpu().numpy()
            gt_envmap_check = gt_envmap.clone().detach().cpu().numpy()
            viewdirs = viewdirs.detach().cpu().numpy()
            save_envmap(None, envmap_check, viewdirs, os.path.join(log_path, 'log_sh_{}'.format(sh_order)), no_npz=True)
            save_envmap(None, gt_envmap_check, viewdirs, os.path.join(log_path, 'log_sh_{}_gt'.format(sh_order)), no_npz=True)
            np.save(os.path.join(log_path, 'sh_{}.npy'.format(sh_order)), envSHs.clone().detach().cpu().numpy())
        if loss.item() < min_loss and step > 2000:
            break
    np.save(sv_path+'.npy', envSHs.clone().detach().cpu().numpy())
    env_map, viewdirs = SH2Envmap(envSHs, H=W, W=W, upper_hemi=False, clamp=False)
    envmap_check = env_map.clone().detach().cpu().numpy()
    viewdirs = viewdirs.detach().cpu().numpy()
    save_envmap(None, envmap_check, viewdirs, sv_path, no_npz=True, no_image=True)
    return envSHs


def image2envmap(img_path, force_white=False):
    im = imageio.imread(img_path)[..., :3] / 255.
    if im.shape[0] > 200:
        scale = im.shape[0] // 200 + 1
        im = cv2.resize(im, (im.shape[1] // scale, im.shape[0] // scale))
        print('Downscale image to: ', im.shape)

    # # y = a * exp(b*x)
    # max_e = 30
    # fixed = .5
    # b = 1 / (fixed - 1) * np.log((fixed + 1) / max_e)
    # a = max_e / (np.exp(b))
    # envmap = a * np.exp(b *im) - 1

    # im = (im - .5) * 2
    # envmap = 10 ** im - 1e-1

    envmap = im ** (1 / 1.24)

    if force_white:
        envmap[..., :] = envmap.mean(axis=-1, keepdims=True)

    return envmap

def save_envmap(envshs, envmap, viewdirs, path, no_npz=False, no_image=False, no_ply=False):
    im = np.power(envmap, 1./2.2)
    im = np.clip(im, 0., 1.)
    im = np.uint8(im * 255.)
    if not no_npz:
        np.save(path + '.npy', envmap)
        if envshs is not None:
            np.save(path + '_shs.npy', envshs)
    if not no_image:
        imageio.imwrite(path + '.png', im)
    if not no_ply:
        if im.shape[-1] != 3:
            im = im.reshape([-1])
            im = np.stack([im, im, im], axis=-1)
        write_ply_rgb(viewdirs.reshape([-1, 3]), im.reshape([-1, 3]), path + '.ply')


if __name__ == '__main__':
    path = './logs/leaf_road/envmap/'
    files = glob.glob(path + 'target*')
    files = [file for file in files if file.endswith('.jpeg') or file.endswith('.png') or file.endswith('.jpg') or file.endswith('JPEG')]
    if len(files) == 0:
        print('No envmap found: ', path)
    file = files[0]
    print('Optimizing SHs towards ', file, ' ...')
    envmap = image2envmap(file, force_white=False)
    pow_num, s = pre_calc_sh_mat()
    envSHs = EnvMap2SH(envmap, sh_order=3, log_path=path+'/log/', min_loss=1e-5, sv_path=path+'_sh', fast=True, pow_num=pow_num.cuda(), s=s.cuda())
