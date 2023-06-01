#imports
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from math import floor, ceil
from random import randint
from sklearn.neighbors import KDTree
from PIL import Image
from sklearn.decomposition import PCA
from skimage.measure import block_reduce


close_threshold = .2
quilt = False  # Warning! For structured textures, quilting may cause deformation. Switch on with un-structured textures.


def transform(data, pca, bounds):
    x = data.reshape([-1, data.shape[-1]])
    x_pca = pca.transform(x)[..., -3:]
    x_bd = (x_pca - bounds[0]) / (bounds[1] - bounds[0])
    x_bd = np.clip(x_bd, 0., 1.)
    x_bd = x_bd.reshape([*data.shape[:-1], x_bd.shape[-1]])
    return x_bd


def get_transform(data, dim):
    x = data.reshape([-1, data.shape[-1]])[..., :dim]
    pca = PCA(n_components=3)
    pca.fit(x)
    x_pca = pca.transform(x)[..., -3:]
    bounds = np.stack([x_pca.min(axis=0), x_pca.max(axis=0)])
    trans_func = lambda a: transform(a[..., :dim], pca=pca, bounds=bounds)
    return trans_func


class patchBasedTextureSynthesis:
    
    def __init__(self, patches, in_outputPath, in_outputSize, in_patchSize, in_overlapSize, in_windowStep = 5, in_mirror_hor = True, in_mirror_vert = True, in_shapshots = True, rotate=True, picked_vertices=None, patch_length=None, coarse_KDtree=True, match_dim=None, sample_tbn=None, strict_match=False, mode='Cut'):

        self.coarse_KDtree = coarse_KDtree
        self.max_patch_res = 32
        self.mode = mode
        self.patches = patches
        self.dim = patches.shape[-1]
        self.match_dim = self.dim if match_dim is None else match_dim

        self.trans_func = get_transform(patches, dim=match_dim)
        self.snapshots = in_shapshots
        self.outputPath = in_outputPath
        self.outputSize = in_outputSize
        self.patchSize = in_patchSize
        self.overlapSize = in_overlapSize
        self.mirror_hor = in_mirror_hor
        self.mirror_vert = in_mirror_vert
        self.rotate = rotate
        self.total_patches_count = 0 #excluding mirrored versions
        self.windowStep = 5
        self.iter = 0
        self.sample_tbn = sample_tbn
        
        self.checkIfDirectoryExists() #check if output directory exists
        self.examplePatches, self.example_tbn = self.prepareExamplePatches()
        self.canvas, self.filledMap, self.idMap = self.initCanvas()
        self.canvas_id = - np.ones(self.canvas.shape[:-1])
        self.initFirstPatch() #place random block to start with
        self.kdtree_topOverlap, self.kdtree_leftOverlap, self.kdtree_combined = self.initKDtrees()

        self.PARM_truncation = 0.  # 5e-2
        self.PARM_attenuation = 1 if not strict_match else 3

        self.picked_vertices = picked_vertices
        self.patch_length = patch_length
        self.cal_dist_matrix()
    
    def cal_dist_matrix(self):
        print('Calculate distance ...')
        if self.picked_vertices is None:
            self.dist = None
            return
        p1 = self.picked_vertices[None]
        p2 = self.picked_vertices[:, None]
        dist = ((p1 - p2)**2).sum(axis=-1) ** .5
        self.dist = dist
        print('Calculation done!')

    def checkIfDirectoryExists(self):
        if not os.path.exists(self.outputPath):
            os.makedirs(self.outputPath)
        
    def resolveAll(self):
        self.saveParams()
        #resolve all unresolved patches
        for i in tqdm(range(np.sum(1-self.filledMap).astype(int))):
            self.resolveNext()
                    
        if not self.snapshots:
            img = Image.fromarray(np.uint8(self.canvas*255))
            img = img.resize((self.outputSize[0], self.outputSize[1]), resample=0, box=None)
            img.save(self.outputPath + 'out.jpg')
        else:
            self.visualize()
        return self.canvas, self.canvas_id
            
    def saveParams(self):
        #write
        text_file = open(self.outputPath + 'params.txt', "w")
        text_file.write("PatchSize: %d \nOverlapSize: %d \nMirror Vert: %d \nMirror Hor: %d" % (self.patchSize, self.overlapSize, self.mirror_vert, self.mirror_hor))
        text_file.close()
        
    def resolveNext(self):
        #coordinate of the next one to resolve
        coord = self.idCoordTo2DCoord(np.sum(self.filledMap), np.shape(self.filledMap)) #get 2D coordinate of next to resolve patch
        #get overlap areas of the patch we want to resolve
        overlapArea_Top = self.getOverlapAreaTop(coord)
        overlapArea_Left = self.getOverlapAreaLeft(coord)
        
        dist = None
        in_k = 16
        while dist is None or dist.shape[0] == 0:
            #find most similar patch from the examples
            dist, ind = self.findMostSimilarPatches(overlapArea_Top, overlapArea_Left, coord, in_k=in_k)
            in_k *= 2
            if self.mirror_hor or self.mirror_vert:
                if self.dist is not None:
                    dist, ind = self.close_patch_check(dist, ind, coord)
                else:
                    #check that top and left neighbours are not mirrors
                    dist, ind = self.checkForMirrors(dist, ind, coord)
        
        #choose random valid patch
        probabilities = self.distances2probability(dist, self.PARM_truncation, self.PARM_attenuation)
        chosenPatchId = np.random.choice(ind, 1, p=probabilities)
        
        #update canvas
        blend_top = (overlapArea_Top is not None)
        blend_left = (overlapArea_Left is not None)
        self.updateCanvas(chosenPatchId, coord[0], coord[1], blend_top, blend_left)
        
        #update filledMap and id map ;)
        self.filledMap[coord[0], coord[1]] = 1
        self.idMap[coord[0], coord[1]] = chosenPatchId
        
        #visualize
        self.visualize()
        
        self.iter += 1
        
    def visualize(self):
        #full visualization includes both example and generated img
        canvasSize = np.shape(self.canvas)
        #insert generated image
        vis = np.zeros((canvasSize[0], canvasSize[1], 3)) + 0.2
        vis[:, 0:canvasSize[1]] = self.trans_func(self.canvas)
        if self.snapshots:
            img = Image.fromarray(np.uint8(vis*255))
            img = img.resize((self.outputSize[0], self.outputSize[1]), resample=0, box=None)
            img.save(self.outputPath + 'out' + str(self.iter) + '.jpg')
                    
    def resize(self, imgArray, targetSize):
        img = Image.fromarray(np.uint8(imgArray*255))
        img = img.resize((targetSize[0], targetSize[1]), resample=0, box=None)
        return np.array(img)/255
        
    def findMostSimilarPatches(self, overlapArea_Top, overlapArea_Left, coord, in_k=16):
        if self.coarse_KDtree:
            overlapArea_Top = self.coarse_top_func(overlapArea_Top[..., :self.match_dim]) if overlapArea_Top is not None else None
            overlapArea_Left = self.coarse_left_func(overlapArea_Left[..., :self.match_dim]) if overlapArea_Left is not None else None

        #check which KD tree we need to use
        if (overlapArea_Top is not None) and (overlapArea_Left is not None):
            combined = self.getCombinedOverlap(overlapArea_Top.reshape(-1), overlapArea_Left.reshape(-1))
            dist, ind = self.kdtree_combined.query([combined], k=in_k)
        elif overlapArea_Top is not None:
            dist, ind = self.kdtree_topOverlap.query([overlapArea_Top.reshape(-1)], k=in_k)
        elif overlapArea_Left is not None:
            dist, ind = self.kdtree_leftOverlap.query([overlapArea_Left.reshape(-1)], k=in_k)
        else:
            raise Exception("ERROR: no valid overlap area is passed to -findMostSimilarPatch-")
        dist = dist[0]
        ind = ind[0]
        return dist, ind
     
    #disallow visually similar blocks to be placed next to each other
    def checkForMirrors(self, dist, ind, coord, thres=1):
        remove_i = []
        #do I have a top or left neighbour
        if coord[0]-1>-1:
            top_neigh = int(self.idMap[coord[0]-1, coord[1]])
            for i in range(len(ind)): 
                if (abs(ind[i]%self.total_patches_count - top_neigh%self.total_patches_count) < thres):
                    remove_i.append(i)
        if  coord[1]-1>-1:
            left_neigh = int(self.idMap[coord[0], coord[1]-1])
            for i in range(len(ind)):
                if (abs(ind[i]%self.total_patches_count - left_neigh%self.total_patches_count) < thres):
                    remove_i.append(i)
        dist = np.delete(dist, remove_i)
        ind = np.delete(ind, remove_i)
        return dist, ind

    # check whether the patches are too close to each other
    def close_patch_check(self, dist, ind, coord, thres=close_threshold):
        remove_i = []
        if coord[0]-1>-1:
            top_neigh = int(self.idMap[coord[0]-1, coord[1]]) % self.total_patches_count
            for i in range(len(ind)): 
                if self.dist[ind[i] % self.total_patches_count, top_neigh] < thres * self.patch_length:
                    remove_i.append(i)
        if  coord[1]-1>-1:
            left_neigh = int(self.idMap[coord[0], coord[1]-1]) % self.total_patches_count
            for i in range(len(ind)):
                if self.dist[ind[i] % self.total_patches_count, left_neigh] < thres * self.patch_length:
                    remove_i.append(i)
        dist = np.delete(dist, remove_i)
        ind = np.delete(ind, remove_i)
        return dist, ind
    
        
    def distances2probability(self, distances, PARM_truncation, PARM_attenuation):

        probabilities = 1 - distances / np.max(distances)  
        probabilities /= np.sum(probabilities) # normalize so they add up to one  
        probabilities *= (probabilities > PARM_truncation)
        probabilities = pow(probabilities, PARM_attenuation) #attenuate the values
        probabilities /= np.sum(probabilities) # normalize so they add up to one  
        if np.isnan(probabilities).any():
            probabilities = np.ones_like(probabilities) / probabilities.shape[0]
        return probabilities
        
    def getOverlapAreaTop(self, coord):
        #do I have a top neighbour
        if coord[0]-1>-1:
            canvasPatch = self.patchCoord2canvasPatch(coord)
            return canvasPatch[0:self.overlapSize, :, :]
        else:
            return None
        
    def getOverlapAreaLeft(self, coord):
        #do I have a left neighbour
        if coord[1]-1>-1:
            canvasPatch = self.patchCoord2canvasPatch(coord)
            return canvasPatch[:, 0:self.overlapSize, :]    
        else:
            return None 
 
    def initKDtrees(self):
        print('Initializing DK tree ...')
        #prepate overlap patches
        topOverlap = self.examplePatches[:, 0:self.overlapSize, :, :self.match_dim]
        leftOverlap = self.examplePatches[:, :, 0:self.overlapSize, :self.match_dim]

        if self.coarse_KDtree:
            block_size = max(int(self.examplePatches.shape[1] / self.max_patch_res), 1)
            self.block_size = block_size
            topOverlap = block_reduce(topOverlap, (1, 1, block_size, 1))
            leftOverlap = block_reduce(leftOverlap, (1, block_size, 1, 1))
            self.coarse_top_func = lambda overlap: block_reduce(overlap, (1, 1, block_size, 1)) if overlap.ndim == 4 else block_reduce(overlap, (1, block_size, 1))
            self.coarse_left_func = lambda overlap: block_reduce(overlap, (1, block_size, 1, 1)) if overlap.ndim == 4 else block_reduce(overlap, (block_size, 1, 1))

        shape_top = np.shape(topOverlap)
        shape_left = np.shape(leftOverlap)
                                   
        flatten_top = topOverlap.reshape(shape_top[0], -1)
        flatten_left = leftOverlap.reshape(shape_left[0], -1)
        flatten_combined = self.getCombinedOverlap(flatten_top, flatten_left) 
        
        tree_top = KDTree(flatten_top)
        tree_left = KDTree(flatten_left)
        tree_combined = KDTree(flatten_combined)
        print('Initialization done!')
        return tree_top, tree_left, tree_combined
    
    #the corner of 2 overlaps is counted double
    def getCombinedOverlap(self, top, left):
        shape = np.shape(top)
        if len(shape) > 1:
            combined = np.zeros((shape[0], shape[1]*2))
            combined[0:shape[0], 0:shape[1]] = top
            combined[0:shape[0], shape[1]:shape[1]*2] = left
        else:
            combined = np.zeros((shape[0]*2))
            combined[0:shape[0]] = top
            combined[shape[0]:shape[0]*2] = left
        return combined

    def initFirstPatch(self):
        #grab a random block 
        patchId = randint(0, np.shape(self.examplePatches)[0])
        #mark out fill map
        self.filledMap[0, 0] = 1
        self.idMap[0, 0] = patchId  # % self.total_patches_count
        #update canvas
        self.updateCanvas(patchId, 0, 0, False, False)
        #visualize
        self.visualize()

        
    def prepareExamplePatches(self):
        print('Preparing example patches ...')
        result = self.patches
        stbn = self.sample_tbn

        self.total_patches_count = result.shape[0]
        if self.mirror_hor:
            hor_result = result[:, ::-1, :, :]
            result = np.concatenate((result, hor_result))
            hor_stbn = np.copy(stbn)
            hor_stbn[..., 0, :] *= -1
            stbn = np.concatenate([stbn, hor_stbn], axis=0)
        if self.mirror_vert:
            vert_result = result[:, :, ::-1, :]
            result = np.concatenate((result, vert_result))
            hor_vtbn = np.copy(stbn)
            hor_vtbn[..., 1, :] *= -1
            stbn = np.concatenate([stbn, hor_vtbn], axis=0)
        if self.rotate:
            rot_result1 = np.rot90(result, 2)
            rot_result2 = np.rot90(rot_result1, 2)
            rot_result3 = np.rot90(rot_result2, 2)
            result = np.concatenate((result, rot_result1, rot_result2, rot_result3))
        return result, stbn
    
    def initCanvas(self):
        
        #check whether the outputSize adheres to patch+overlap size
        num_patches_X = ceil((self.outputSize[0]-self.overlapSize)/(self.patchSize+self.overlapSize))
        num_patches_Y = ceil((self.outputSize[1]-self.overlapSize)/(self.patchSize+self.overlapSize))
        #calc needed output image size
        required_size_X = num_patches_X*self.patchSize + (num_patches_X+1)*self.overlapSize
        required_size_Y = num_patches_Y*self.patchSize + (num_patches_X+1)*self.overlapSize
        
        #create empty canvas
        canvas = np.zeros((required_size_X, required_size_Y, self.dim))
        filledMap = np.zeros((num_patches_X, num_patches_Y)) #map showing which patches have been resolved
        idMap = np.zeros((num_patches_X, num_patches_Y)) - 1 #stores patches id
        
        print("modified output size: ", np.shape(canvas))
        print("number of patches: ", np.shape(filledMap)[0])

        return canvas, filledMap, idMap

    def idCoordTo2DCoord(self, idCoord, imgSize):
        row = int(floor(idCoord / imgSize[0]))
        col = int(idCoord - row * imgSize[1])
        return [row, col]

    def updateCanvas(self, inputPatchId, coord_X, coord_Y, blendTop = False, blendLeft = False, mode=None):
        mode = self.mode if mode is None else mode
        #translate Patch coordinate into Canvas coordinate
        x_range = self.patchCoord2canvasCoord(coord_X)
        y_range = self.patchCoord2canvasCoord(coord_Y)
        examplePatch = self.examplePatches[inputPatchId]
        examplePatch_id = inputPatchId * np.ones_like(examplePatch[..., 0])
        if blendLeft:
            canvasOverlap_id = self.canvas_id[x_range[0]:x_range[1], y_range[0]:y_range[0]+self.overlapSize]
            examplePatchOverlap_id = np.copy(examplePatch_id[0][:, 0:self.overlapSize])
            canvasOverlap = self.canvas[x_range[0]:x_range[1], y_range[0]:y_range[0]+self.overlapSize]
            examplePatchOverlap = np.copy(examplePatch[0][:, 0:self.overlapSize])
            if mode == 'Cut':
                examplePatch[0][:, 0:self.overlapSize], mask = self.MinErrBouCut(canvasOverlap, examplePatchOverlap)
                examplePatch_id[0][:, 0:self.overlapSize] = np.where(mask[..., 0], canvasOverlap_id, examplePatchOverlap_id)
            else:
                examplePatch[0][:, 0:self.overlapSize], mask = self.linearBlendOverlaps(canvasOverlap, examplePatchOverlap, 'left')
                examplePatch_id[0][:, 0:self.overlapSize] = np.where(mask[..., 0], canvasOverlap_id, examplePatchOverlap_id)
        if blendTop:
            canvasOverlap_id = self.canvas_id[x_range[0]:x_range[0]+self.overlapSize, y_range[0]:y_range[1]]
            examplePatchOverlap_id = np.copy(examplePatch_id[0][0:self.overlapSize, :])
            canvasOverlap = self.canvas[x_range[0]:x_range[0]+self.overlapSize, y_range[0]:y_range[1]]
            examplePatchOverlap = np.copy(examplePatch[0][0:self.overlapSize, :])
            if mode == 'Cut':
                out, mask = self.MinErrBouCut(np.moveaxis(canvasOverlap, 0, 1), np.moveaxis(examplePatchOverlap, 0, 1))
                examplePatch[0][:self.overlapSize, :] = np.moveaxis(out, 0, 1)
                examplePatch_id[0][:self.overlapSize] = np.where(np.moveaxis(mask[..., 0], 0, 1), canvasOverlap_id, examplePatchOverlap_id)
            else:
                examplePatch[0][0:self.overlapSize, :], mask = self.linearBlendOverlaps(canvasOverlap, examplePatchOverlap, 'top')
                examplePatch_id[0][:self.overlapSize] = np.where(mask[..., 0], canvasOverlap_id, examplePatchOverlap_id)
        self.canvas[x_range[0]:x_range[1], y_range[0]:y_range[1]] = examplePatch
        self.canvas_id[x_range[0]:x_range[1], y_range[0]:y_range[1]] = examplePatch_id
        
    def linearBlendOverlaps(self, canvasOverlap, examplePatchOverlap, mode):
        if mode == 'left':
            mask = np.repeat(np.arange(self.overlapSize)[np.newaxis, :], np.shape(canvasOverlap)[0], axis=0) / self.overlapSize
        elif mode == 'top':
            mask = np.repeat(np.arange(self.overlapSize)[:, np.newaxis], np.shape(canvasOverlap)[1], axis=1) / self.overlapSize
        mask = np.repeat(mask[:, :, np.newaxis], self.dim, axis=2)
        mask[..., self.match_dim:] = np.array(mask[..., self.match_dim:] > .5, np.float32)
        return canvasOverlap * (1 - mask) + examplePatchOverlap * mask, np.broadcast_to(np.array(mask < .5, dtype=bool), canvasOverlap.shape)
    
    def MinErrBouCut(self, B1, B2):
        H, W = B1.shape[:2]

        # Dynamic Programming
        e = ((B1[..., :self.match_dim] - B2[..., :self.match_dim]) ** 2).sum(axis=-1)
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
        mask = np.zeros_like(B1, dtype=bool)
        for i in range(H):
            if not quilt:
                B[i, :trace[i]] = B1[i, :trace[i]]
                B[i, trace[i]:] = B2[i, trace[i]:]
                B[i, trace[i]] = (B1[i, trace[i]] + B2[i, trace[i]] ) / 2
            else:
                weight1 = np.linspace(1., .5, trace[i])
                weight2 = np.linspace(.5, 1., B.shape[1] - trace[i])
                B[i, :trace[i]] = B1[i, :trace[i]] * weight1[..., None] + B2[i, :trace[i]] * (1 - weight1[..., None])
                B[i, trace[i]:] = B2[i, trace[i]:] * weight2[..., None] + B1[i, trace[i]:] * (1 - weight2[..., None])

            mask[i, :trace[i]] = True

        return B, mask
    
    def patchCoord2canvasCoord(self, coord):
        return [(self.patchSize+self.overlapSize)*coord, (self.patchSize+self.overlapSize)*(coord+1) + self.overlapSize]
    
    def patchCoord2canvasPatch(self, coord):
        x_range = self.patchCoord2canvasCoord(coord[0])
        y_range = self.patchCoord2canvasCoord(coord[1])
        return np.copy(self.canvas[x_range[0]:x_range[1], y_range[0]:y_range[1]])
    

if __name__ == '__main__':
    #PARAMS
    ########################################################################################
    # if not sure just check the logs directory of NeRF-Texture
    DATA_NAME = 'test'
    MODEL_NAME = 'curved_grid_hash_clus_optcam_SH'
    data_path = f'./logs/{DATA_NAME}/field/'
    save_path = data_path
    ########################################################################################
    data = np.load(f'{data_path}/{MODEL_NAME}.npz', allow_pickle=True)
    patches = data['patches']
    patch_idx = np.arange(patches.shape[0])  #[::4]
    patches = patches[patch_idx]
    grid_gap = data['grid_gap']
    match_dim = data['patches'].shape[-1]

    if 'patch_phi_embed' in data.keys() and data['patch_phi_embed'].ndim > 0:
        phi_embed_dim = data['patch_phi_embed'].shape[-1]
        patches = np.concatenate([patches, data['patch_phi_embed'][patch_idx]], axis=-1)
    else:
        phi_embed_dim = 0

    if 'patch_local_tbn' in data.keys() and data['patch_local_tbn'].ndim > 0:
        patches = np.concatenate([patches, data['patch_local_tbn'][patch_idx]], axis=-1)
    
    patch_length = patches.shape[1] * grid_gap
    picked_vertices = data['picked_vertices'][patch_idx]

    print('Patches shape: ', patches.shape)
    B, H, W = patches.shape[:3]

    # Options
    #######################################################################################
    outputSize = [1024*2, 1024*2]
    mode = 'Cut'
    # mode = 'blend'  # it looks ok at most cases but overall worse than 'Cut'
    patchSize = int(H/4)  # size of the patch (without the overlap), could also be int(5*H/7) for acceleration
    strict_match = True  # could switch to False for diverse synthesis
    in_mirror_vert = False  # could set to True for more augmented patches
    in_mirror_hor = False
    #######################################################################################

    outputPath = f"{save_path}/{DATA_NAME}_{MODEL_NAME}/"
    if (H - patchSize) % 2 == 1:
        patchSize -= 1
    overlapSize = int((H-patchSize)/2) #the width of the overlap region

    pbts = patchBasedTextureSynthesis(patches, outputPath, outputSize, patchSize, overlapSize, in_windowStep = 5, in_mirror_hor = in_mirror_hor, in_mirror_vert = in_mirror_vert, rotate=False, in_shapshots=True, picked_vertices=picked_vertices, patch_length=patch_length, coarse_KDtree=True, sample_tbn=data['patch_sample_tbn'], match_dim=match_dim, strict_match=strict_match, mode=mode)
    canvas, canvas_id = pbts.resolveAll()

    canvas_id = np.array(canvas_id, dtype=np.int32)
    total_id = np.sort(np.unique(canvas_id.reshape([-1])))

    index_dict = {}
    for i in range(total_id.shape[0]):
        index_dict[total_id[i]] = i

    total_id_vis = np.sort(np.unique(canvas_id.reshape([-1]) % pbts.total_patches_count))
    index_dict_vis = {}
    for i in range(total_id_vis.shape[0]):
        index_dict_vis[total_id_vis[i]] = i
    
    canvas_id_vis = canvas_id  % pbts.total_patches_count
    for i in range(canvas_id.shape[0]):
        for j in range(canvas_id.shape[1]):
            canvas_id[i, j] = index_dict[canvas_id[i, j]]
            canvas_id_vis[i, j] = index_dict_vis[canvas_id_vis[i, j]]
    cmap = plt.cm.get_cmap('cubehelix', canvas_id_vis.max())
    canvas_id_vis = cmap(canvas_id_vis)
    Image.fromarray(np.array(255 * canvas_id_vis, dtype=np.uint8)).save(outputPath + '/patch_id.png')
    
    sample_tbn = pbts.example_tbn[total_id]
    features = pbts.canvas[..., :match_dim]
    phi_embed = pbts.canvas[..., match_dim:match_dim+phi_embed_dim] if 'patch_phi_embed' in data.keys() else None
    local_tbn = pbts.canvas[..., -9:] if 'patch_local_tbn' in data.keys() else None

    np.savez(pbts.outputPath + '/../texture.npz', features=pbts.canvas[..., :match_dim], mesh=None, grid_gap=grid_gap, sample_tbn=sample_tbn, sample_tbn_ids=canvas_id, phi_embed=phi_embed, local_tbn=local_tbn)
    print('Finish')
