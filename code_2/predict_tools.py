# https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/blob/ba50a82dce412df97f088c572d86d7977753bf74/lib/core/inference.py#L18:5
# https://www.dacon.io/competitions/official/235701/codeshare/2478?page=1&dtype=recent
import numpy as np
import cv2
from numpy.linalg import LinAlgError
import torch

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals
# def get_max_preds(batch_heatmaps):
# 	batch_size = batch_heatmaps.shape[0]
# 	num_joints = batch_heatmaps.shape[1]
# 	width      = batch_heatmaps.shape[3]

# 	heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
# 	idx               = np.argmax(heatmaps_reshaped, 2)
# 	maxvals           = np.amax(heatmaps_reshaped, 2)

# 	maxvals = maxvals.reshape((batch_size, num_joints, 1))
# 	idx     = idx.reshape((batch_size, num_joints, 1))

# 	preds   = np.tile(idx, (1,1,2)).astype(np.float32)

# 	preds[:,:,0] = (preds[:,:,0]) % width
# 	preds[:,:,1] = np.floor((preds[:,:,1]) / width)

# 	pred_mask    = np.tile(np.greater(maxvals, 0.0), (1,1,2))
# 	pred_mask    = pred_mask.astype(np.float32)

# 	preds *= pred_mask
# 	return preds, maxvals


def dark_post_processing(coords, batch_heatmaps):
    '''
    DARK post-pocessing
    :param coords: batchsize*num_kps*2
    :param batch_heatmaps:batchsize*num_kps*high*width
    :return:
    '''

    shape_pad = list(batch_heatmaps.shape)
    shape_pad[2] = shape_pad[2] + 2
    shape_pad[3] = shape_pad[3] + 2

    for i in range(shape_pad[0]):
        for j in range(shape_pad[1]):
            mapij = batch_heatmaps[i, j, :, :]
            maxori = np.max(mapij)
            mapij = cv2.GaussianBlur(mapij, (7, 7), 0)
            max = np.max(mapij)
            min = np.min(mapij)
            mapij = (mapij-min)/(max-min) * maxori
            batch_heatmaps[i, j, :, :] = mapij
    batch_heatmaps = np.clip(batch_heatmaps, 0.001, 50)
    batch_heatmaps = np.log(batch_heatmaps)
    batch_heatmaps_pad = np.zeros(shape_pad, dtype=float)
    batch_heatmaps_pad[:, :, 1:-1, 1:-1] = batch_heatmaps
    batch_heatmaps_pad[:, :, 1:-1, -1] = batch_heatmaps[:, :, :, -1]
    batch_heatmaps_pad[:, :, -1, 1:-1] = batch_heatmaps[:, :, -1, :]
    batch_heatmaps_pad[:, :, 1:-1, 0] = batch_heatmaps[:, :, :, 0]
    batch_heatmaps_pad[:, :, 0, 1:-1] = batch_heatmaps[:, :, 0, :]
    batch_heatmaps_pad[:, :, -1, -1] = batch_heatmaps[:, :, -1, -1]
    batch_heatmaps_pad[:, :, 0, 0] = batch_heatmaps[:, :, 0, 0]
    batch_heatmaps_pad[:, :, 0, -1] = batch_heatmaps[:, :, 0, -1]
    batch_heatmaps_pad[:, :, -1, 0] = batch_heatmaps[:, :, -1, 0]
    I = np.zeros((shape_pad[0], shape_pad[1]))
    Ix1 = np.zeros((shape_pad[0], shape_pad[1]))
    Iy1 = np.zeros((shape_pad[0], shape_pad[1]))
    Ix1y1 = np.zeros((shape_pad[0], shape_pad[1]))
    Ix1_y1_ = np.zeros((shape_pad[0], shape_pad[1]))
    Ix1_ = np.zeros((shape_pad[0], shape_pad[1]))
    Iy1_ = np.zeros((shape_pad[0], shape_pad[1]))
    coords = coords.astype(np.int32)
    for i in range(shape_pad[0]):
        for j in range(shape_pad[1]):
            I[i, j] = batch_heatmaps_pad[i, j,
                                         coords[i, j, 1]+1, coords[i, j, 0]+1]
            Ix1[i, j] = batch_heatmaps_pad[i, j,
                                           coords[i, j, 1]+1, coords[i, j, 0] + 2]
            Ix1_[i, j] = batch_heatmaps_pad[i, j,
                                            coords[i, j, 1]+1, coords[i, j, 0]]
            Iy1[i, j] = batch_heatmaps_pad[i, j,
                                           coords[i, j, 1] + 2, coords[i, j, 0]+1]
            Iy1_[i, j] = batch_heatmaps_pad[i, j,
                                            coords[i, j, 1], coords[i, j, 0]+1]
            Ix1y1[i, j] = batch_heatmaps_pad[i, j,
                                             coords[i, j, 1] + 2, coords[i, j, 0] + 2]
            Ix1_y1_[i, j] = batch_heatmaps_pad[i,
                                               j, coords[i, j, 1], coords[i, j, 0]]
    dx = 0.5 * (Ix1 - Ix1_)
    dy = 0.5 * (Iy1 - Iy1_)
    D = np.zeros((shape_pad[0], shape_pad[1], 2))
    D[:, :, 0] = dx
    D[:, :, 1] = dy
    D.reshape((shape_pad[0], shape_pad[1], 2, 1))
    dxx = Ix1 - 2*I + Ix1_
    dyy = Iy1 - 2*I + Iy1_
    dxy = 0.5*(Ix1y1 - Ix1 - Iy1 + I + I - Ix1_-Iy1_+Ix1_y1_)
    hessian = np.zeros((shape_pad[0], shape_pad[1], 2, 2))
    hessian[:, :, 0, 0] = dxx
    hessian[:, :, 1, 0] = dxy
    hessian[:, :, 0, 1] = dxy
    hessian[:, :, 1, 1] = dyy
    inv_hessian = np.zeros(hessian.shape)
    # hessian_test = np.zeros(hessian.shape)
    for i in range(shape_pad[0]):
        for j in range(shape_pad[1]):
            hessian_tmp = hessian[i, j, :, :]
            try:
                inv_hessian[i, j, :, :] = np.linalg.inv(hessian_tmp)
            except LinAlgError:
                inv_hessian[i, j, :, :] = np.zeros((2, 2))
            # hessian_test[i,j,:,:] = np.matmul(hessian[i,j,:,:],inv_hessian[i,j,:,:])
            # print( hessian_test[i,j,:,:])
    res = np.zeros(coords.shape)
    coords = coords.astype(np.float)
    for i in range(shape_pad[0]):
        for j in range(shape_pad[1]):
            D_tmp = D[i, j, :]
            D_tmp = D_tmp[:, np.newaxis]
            shift = np.matmul(inv_hessian[i, j, :, :], D_tmp)
            # print(shift.shape)
            res_tmp = coords[i, j, :] - shift.reshape((-1))
            res[i, j, :] = res_tmp
    return res


def calc_coord_loss(pred, gt):
    batch_size = gt.size(0)
    valid_mask = gt[:, :, -1].view(batch_size, -1, 1)
    gt = gt[:, :, :2]
    return torch.mean(torch.sum(torch.abs(pred-gt) * valid_mask, dim=-1))

def get_final_preds(cfg, batch_heatmaps):
    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]
    if cfg.target_type == 'gaussian':
        coords, maxvals = get_max_preds(batch_heatmaps)
        if cfg.post_processing == "dark":
            coords = dark_post_processing(coords, batch_heatmaps)
    elif cfg.target_type == 'offset':
        net_output = batch_heatmaps.copy()
        kps_pos_distance_x = cfg.kpd
        kps_pos_distance_y = cfg.kpd
        batch_heatmaps = net_output[:, ::3, :]
        offset_x = net_output[:, 1::3, :] * kps_pos_distance_x
        offset_y = net_output[:, 2::3, :] * kps_pos_distance_y
        for i in range(batch_heatmaps.shape[0]):
            for j in range(batch_heatmaps.shape[1]):
                batch_heatmaps[i, j, :, :] = cv2.GaussianBlur(
                    batch_heatmaps[i, j, :, :], (15, 15), 0)
                offset_x[i, j, :, :] = cv2.GaussianBlur(
                    offset_x[i, j, :, :], (7, 7), 0)
                offset_y[i, j, :, :] = cv2.GaussianBlur(
                    offset_y[i, j, :, :], (7, 7), 0)
        coords, maxvals = get_max_preds(batch_heatmaps)
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                px = int(coords[n][p][0])
                py = int(coords[n][p][1])
                coords[n][p][0] += offset_x[n, p, py, px]
                coords[n][p][1] += offset_y[n, p, py, px]

    preds = coords.copy()
    preds[:, :, 0] = preds[:, :, 0] / \
        (heatmap_width - 1.0) * (4 * heatmap_width - 1.0)
    preds[:, :, 1] = preds[:, :, 1] / \
        (heatmap_height - 1.0) * (4 * heatmap_height - 1.0)

    return preds


# https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/blob/ba50a82dce412df97f088c572d86d7977753bf74/lib/core/evaluate.py#L41
# https://www.dacon.io/competitions/official/235701/codeshare/2478?page=1&dtype=recent
# 수정 필요
def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


def accuracy_heatmap(output, target, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    dists = calc_dists(pred, target, norm)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred


def accuracy(output, target, thr_PCK, thr_PCKh, dataset, hm_type='gaussian', threshold=0.5):
	idx = list(range(output.shape[1]))
	norm = 1.0

	if hm_type == 'gaussian':
		pred, _ = get_max_preds(output)
		target, _ = get_max_preds(target)

		h = output.shape[2]
		w = output.shape[3]
		norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10

	dists = calc_dists(pred, target, norm)

	acc = np.zeros((len(idx)))
	avg_acc = 0
	cnt = 0
	visible = np.zeros((len(idx)))

	for i in range(len(idx)):
		acc[i] = dist_acc(dists[idx[i]])
		if acc[i] >= 0:
			avg_acc = avg_acc + acc[i]
			cnt += 1
			visible[i] = 1
		else:
			acc[i] = 0

	avg_acc = avg_acc / cnt if cnt != 0 else 0

	if cnt != 0:
		acc[0] = avg_acc

	# PCKh
	PCKh = np.zeros((len(idx)))
	avg_PCKh = 0
	headLength = np.linalg.norm(target[0, 1, :] - target[0, 2, :])

	for i in range(len(idx)):
		PCKh[i] = dist_acc(dists[idx[i]], thr_PCKh*headLength)
		if PCKh[i] >= 0:
			avg_PCKh = avg_PCKh + PCKh[i]
		else:
			PCKh[i] = 0

	avg_PCKh = avg_PCKh / cnt if cnt != 0 else 0

	if cnt != 0:
		PCKh[0] = avg_PCKh

	# PCK
	PCK = np.zeros((len(idx)))
	avg_PCK = 0

	torso = np.linalg.norm(target[0, 3, :] - target[0, 10, :])

	for i in range(len(idx)):
		PCK[i] = dist_acc(dists[idx[i]], thr_PCK*torso)

		if PCK[i] >= 0:
			avg_PCK = avg_PCK + PCK[i]
		else:
			PCK[i] = 0

	avg_PCK = avg_PCK / cnt if cnt != 0 else 0

	if cnt != 0:
		PCK[0] = avg_PCK

	return acc, PCK, PCKh, cnt, pred, visible
