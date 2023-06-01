import numpy as np
import timeit
import cv2
import matplotlib.pyplot as plt
from numba import njit


def MinErrBouCut(B1, B2):
    H, W = B1.shape[:2]

    # Dynamic Programming
    e = ((B1 - B2) ** 2).sum(axis=-1)
    E = np.zeros_like(e)
    E[0, :] = e[0, :]
    T = np.zeros_like(e, dtype=np.int32)
    T[0, :] = np.arange(T.shape[-1])
    for i in range(1, H):
        for j in range(W):
            jrange = np.arange(max(0, j-1), min(j+1, W-1)+1)
            j_ = np.argmin(E[i-1, jrange])
            E[i, j] = e[i, j] + E[i-1, jrange][j_]
            T[i, j] = jrange[j_]
    
    # Trace Back
    dest = np.argmin(E[-1])
    trace = np.zeros([H], dtype=np.int32)
    trace[-1] = dest
    for i in range(2, H+1):
        trace[-i] = T[-i+1, trace[-i+1]]
    
    # Quilting
    B = np.zeros_like(B1)
    for i in range(H):
        B[i, :trace[i]] = B1[i, :trace[i]]
        B[i, trace[i]:] = B2[i, trace[i]:]
        B[i, trace[i]] = (B1[i, trace[i]] + B2[i, trace[i]] ) / 2

    return B, trace


def FloydCut_naive(B1, B2):
    H, W = B1.shape[:2]
    N = H * W + 2

    def idx2coor(idx, inverse=False):
        if not inverse:
            h, w = idx // W, idx % W
            return np.stack([h, w], axis=-1)
        else:
            return idx[..., 0] * W + idx[..., 1]
    
    def neibor(coor):
        x, y = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2), indexing='ij')
        xy = np.stack([x, y], axis=-1).reshape([-1, 2])
        xy = np.delete(xy, 4, 0)
        nb = coor + xy

        mask = (nb >= 0).all(axis=-1)
        mask = np.logical_and(mask, nb[..., 0] < H )
        mask = np.logical_and(mask, nb[..., 1] < W )
        nb = nb[mask]

        return nb

    # Initialization
    e = ((B1 - B2) ** 2).sum(axis=-1)
    dist = np.inf * np.ones([N, N], dtype=np.float32)
    for i in range(N-2):
        dist[i, i] = 0
        coor = idx2coor(i)
        d = e[coor[0], coor[1]]
        nb = neibor(coor)
        nb_idx = idx2coor(nb, True)
        dist[nb_idx, i] = d
    dist[-1, -1] = dist[-2, -2] = 0
    row0_idx = idx2coor(np.stack([np.zeros([W], dtype=np.int32), np.arange(W)], axis=-1), True)
    rowm1_idx = idx2coor(np.stack([(H - 1) * np.ones([W], dtype=np.int32), np.arange(W)], axis=-1), True)
    dist[-2, row0_idx] = e[0]
    dist[rowm1_idx, -1] = 0
    # Trace
    traces = []
    for i in range(N):
        traces.append([])
        for j in range(N):
            traces[i].append([])

    # Dynamic Programming
    for k in range(N):
        for i in range(N):
            for j in range(N):
                if dist[i, j] > dist[i, k] + dist[k, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]
                    traces[i][j] = traces[i][k] + [idx2coor(k)] + traces[k][j]
    
    trace = np.stack(traces[N-2][N-1], axis=0)
    
    # Quilting
    flag = np.zeros_like(B1[..., 0], dtype=bool)
    def dfs(coor):
        if (coor < 0).any() or coor[0] >= flag.shape[0] or coor[1] >= flag.shape[1]:
            return
        if (coor == trace).all(axis=-1).any() or flag[coor[0], coor[1]]:
            return
        else:
            flag[coor[0], coor[1]] = True
            dfs(coor+np.array([-1, 0]))
            dfs(coor+np.array([1, 0]))
            dfs(coor+np.array([0, -1]))
            dfs(coor+np.array([0, 1]))
    for i in range(H):
        if not (np.array([i, 0]) == trace).all(-1).any():
            dfs(np.array([i, 0]))
    B = np.where(flag[..., None], B1, B2)
    # Blend
    for i in range(trace.shape[0]):
        B[trace[i, 0], trace[i, 1]] = (B1[trace[i, 0], trace[i, 1]] + B2[trace[i, 0], trace[i, 1]]) / 2
    
    return B, trace


def trace_back(dist, H, W):
    # from test import FloydCut as fcut
    # b, t, d = fcut(B1, B2)

    def neibor(coor):
        x, y = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2), indexing='ij')
        xy = np.stack([x, y], axis=-1).reshape([-1, 2])
        xy = np.delete(xy, 4, 0)
        nb = coor + xy

        mask = (nb >= 0).all(axis=-1)
        mask = np.logical_and(mask, nb[..., 0] < H )
        mask = np.logical_and(mask, nb[..., 1] < W )
        nb = nb[mask]

        return nb

    def idx2coor(idx, inverse=False):
        if not inverse:
            h, w = idx // W, idx % W
            return np.stack([h, w], axis=-1)
        else:
            return idx[..., 0] * W + idx[..., 1]
    
    # Pick H=0 starting point
    epsilon = 1e-3
    flag = False
    while not flag:
        idxs = np.where((dist[-2, :] + dist[:, -1] - dist[-2, -1]) < epsilon)[0]
        epsilon *= 2
        start_idx = np.min(idxs)
        # assert start_idx < W, 'Start point not in 1st row, Dist wrong !'
        flag = start_idx < W
        if not flag:
            print('Reduce Epsilon (start_idx check)')
            continue

        trace = np.array([idx2coor(start_idx)], dtype=np.int32)
        def recursive_trace(trace):
            current_idx = idx2coor(trace[-1:], True)[0]
            # Check whether in shortest path
            if current_idx not in idxs:
                return trace[:-1]
            # Check Loop
            if trace.shape[0] > 1:
                trace_idx = idx2coor(trace[:-1], True)
                if current_idx in trace_idx:
                    return trace[:-1]
            # Check terminate
            if trace[-1, 0] == H - 1:
                return trace
            # Recursion
            nb = neibor(trace[-1])
            for i in range(nb.shape[0]):
                trace = np.concatenate([trace, nb[i:i+1]])
                trace = recursive_trace(trace)
                if trace[-1, 0] == H - 1:
                    return trace
            # print('No neighbours in shortest path... Dist may be wrong')
            # print('Current position: ', trace[-1], '\nNeighbours: \n', nb)
            # print('Shortest Path: \n', idx2coor(idxs))
            return trace[:-1]
        trace = recursive_trace(trace)
        if len(trace) == 0:
            print('Reduce Epsilon (trace back check)')
            flag = False

    return trace


@njit
def floyd(dist):
    N = dist.shape[0]
    # Dynamic Programming
    for k in range(N):
        for i in range(N):
            for j in range(N):
                if dist[i, j] > dist[i, k] + dist[k, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]
    return dist


def inflection_detection(trace):
    inflection = np.zeros([trace.shape[0]], dtype=bool)
    left, right = 0, 1
    while left < inflection.shape[0]:
        while right < inflection.shape[0] and trace[right, 0] == trace[left, 0]:
            right += 1
        if right == inflection.shape[0]:
            break
        if left > 0 and trace[left-1, 0] == trace[right, 0]:
            inflection[left: right] = True
        left = right
        right += 1
    return inflection


def FloydCut(B1, B2):

    H, W = B1.shape[:2]
    N = H * W + 2

    def idx2coor(idx, inverse=False):
        if not inverse:
            h, w = idx // W, idx % W
            return np.stack([h, w], axis=-1)
        else:
            return idx[..., 0] * W + idx[..., 1]
    
    def neibor(coor):
        x, y = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2), indexing='ij')
        xy = np.stack([x, y], axis=-1).reshape([-1, 2])
        xy = np.delete(xy, 4, 0)
        nb = coor + xy

        mask = (nb >= 0).all(axis=-1)
        mask = np.logical_and(mask, nb[..., 0] < H )
        mask = np.logical_and(mask, nb[..., 1] < W )
        nb = nb[mask]

        return nb

    # Initialization
    e = ((B1 - B2) ** 2).sum(axis=-1)
    dist = np.inf * np.ones([N, N], dtype=np.float32)
    for i in range(N-2):
        dist[i, i] = 0
        coor = idx2coor(i)
        d = e[coor[0], coor[1]]
        nb = neibor(coor)
        nb_idx = idx2coor(nb, True)
        dist[nb_idx, i] = d
    dist[-1, -1] = 0
    dist[-2, -2] = 0
    row0_idx = idx2coor(np.stack([np.zeros([W], dtype=np.int32), np.arange(W)], axis=-1), True)
    rowm1_idx = idx2coor(np.stack([(H - 1) * np.ones([W], dtype=np.int32), np.arange(W)], axis=-1), True)
    dist[-2, row0_idx] = e[0]
    dist[rowm1_idx, -1] = 0

    # Dynamic Programming
    dist = floyd(dist)

    # Tracing Back
    trace = trace_back(dist, H, W)

    # Inflection Mask
    inflection = inflection_detection(trace)

    # Quilting
    flag = np.zeros_like(B1[..., 0], dtype=bool)
    for i in range(H):
        cross_pts = np.sort(trace[trace[:, 0]==i, 1])
        cross_pts = np.concatenate([[0], cross_pts])
        value = True
        j = 0
        while j < cross_pts.shape[0]-1:
            flag[i, cross_pts[j]:cross_pts[j+1]] = value
            # Check inflection
            jid = np.where(np.absolute(trace - np.array([i, cross_pts[j+1]])).sum(-1)==0)[0][0]
            if not inflection[jid]:
                value = not value

            j += 1
            while cross_pts[j] + 1 in cross_pts:
                # Pass horizonal line in trace
                j += 1

    B = np.where(flag[..., None], B1, B2)
    # Blend
    for i in range(trace.shape[0]):
        B[trace[i, 0], trace[i, 1]] = (B1[trace[i, 0], trace[i, 1]] + B2[trace[i, 0], trace[i, 1]]) / 2
    
    return B, trace


if __name__ == '__main__':
    
    test_opt_version = True

    for _ in range(100):
        if test_opt_version:
            H, W = 50, 30
            N = H * W + 2
            B1, B2 = np.random.rand(H, W, 3), np.random.rand(H, W, 3)
            B1[..., 1:] *= 0.8
            B2[..., :-1] *= .8
            
            x, y = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2), indexing='ij')
            xy = np.stack([x, y], axis=-1).reshape([-1, 2])
            xy = np.delete(xy, 4, 0)

            for _ in range(3):
                B1 = cv2.GaussianBlur(B1, (3, 3), 0)
                B2 = cv2.GaussianBlur(B2, (3, 3), 0)

            for i in range(W-2):
                B1[i+5:i+7, i] = -np.inf
                B1[-i-5:-i-3, -i-1] = -np.inf
            
            # j = 0
            # js = []
            # for i in range(H):
            #     jrange = np.arange(max(0, j-1), min(j+1, W-1)+1)
            #     j = np.random.choice(jrange)
            #     js.append([i, j])
            #     B1[i, j] = B2[i, j]
            # js = np.array(js)
                            
            start = timeit.default_timer()
            B, trace = FloydCut(B1, B2)
            end = timeit.default_timer()
            print(end - start, 's')

            # print((trace==js).all())

            B_ = np.copy(B)
            inflection = inflection_detection(trace)
            for i in range(trace.shape[0]):
                B_[trace[i, 0], trace[i, 1]] = 1. if not inflection[i] else np.array([1., 0., 0.])
            image = np.concatenate([B1, B2, B, B_], axis=1)
            plt.imshow(image)
            plt.show()

        else:
            H, W = 30, 10
            B1, B2 = np.random.rand(H, W, 3), np.random.rand(H, W, 3)
            for _ in range(3):
                B1 = cv2.GaussianBlur(B1, (3, 3), 0)
                B2 = cv2.GaussianBlur(B2, (3, 3), 0)

            for i in range(W-1):
                B1[i+5:i+7, i] = -np.inf
                B1[i+15:i+17, i+1] = -np.inf

            start = timeit.default_timer()
            B, trace = FloydCut_naive(B1, B2)
            end = timeit.default_timer()
            print(end - start, 's')

            B_ = np.copy(B)
            for i in range(trace.shape[0]):
                B_[trace[i, 0], trace[i, 1]] = 1.
            image = np.concatenate([B1, B2, B, B_], axis=1)
            plt.imshow(image)
            plt.show()
