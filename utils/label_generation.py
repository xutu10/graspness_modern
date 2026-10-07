import torch

from utils.knn_utils import knn_query
from utils.loss_utils import GRASP_MAX_WIDTH, batch_viewpoint_params_to_matrix, \
    transform_point_cloud, generate_grasp_views

VIEW_U_THRESHOLD = 0.6
VIEW_GRASP_NUM = 48  # A * D normalisation constant used by the original code


def _remap_views(pose_rot, template_views, template_rot):
    """For one object: match each template view to the closest rotated view and
    return (view_inds (V,), rotations (V, 3, 3)) expressed in the camera frame."""
    rotated_views = transform_point_cloud(template_views, pose_rot, '3x3') #(300, 3)
    # find nearest rotated view for each template view and return index
    view_inds = knn_query(rotated_views, k=1, query_pos=template_views).reshape(-1)
    # compute the rotation of each matched view in camera coordinates and reorder index
    rots = (pose_rot @ template_rot)[view_inds] # (300, 3, 3)
    return view_inds, rots


def _collect_sample(end_points, b, seed_xyz, template_views, template_rot, has_stable):
    """Build per-seed-point labels for one sample b, selecting the Ns rows per object
    before anything large is concatenated. Does not modify end_points."""
    poses = end_points['object_poses_list'][b]
    offset = end_points['cloud_offset'][b]
    scores_list = end_points['grasp_scores_list'][b]    # num_obj * (Np_k, V, A, D)
    widths_list = end_points['grasp_widths_list'][b]    # num_obj * (Np_k, V, A, D)
    stable_list = end_points['grasp_stable_list'][b] if has_stable else None  # num_obj * (Np_k, V, A)

    # per object: transformed cloud points and view remapping
    transformed_pts, view_maps, view_rots = [], [], []
    for k, pose in enumerate(poses):
        pts = transform_point_cloud(end_points['grasp_points_list'][b][k], pose, '3x4')
        transformed_pts.append(pts + offset)                                  # (Np_k, 3)
        inds, rots = _remap_views(pose[:3, :3], template_views, template_rot)
        view_maps.append(inds)                                           # (V,)
        view_rots.append(rots)                                           # (V, 3, 3)

    # nearest cloud point for every seed point 
    all_pts = torch.cat(transformed_pts, dim=0)                               # (Np', 3)
    seed_idx = knn_query(all_pts, k=1, query_pos=seed_xyz).reshape(-1)   # (Ns,) among all Np points

    # create output tensors with shape
    Ns = seed_xyz.size(0)
    _, V, A, D = scores_list[0].shape
    out_scores = scores_list[0].new_zeros((Ns, V, A, D))
    out_widths = widths_list[0].new_zeros((Ns, V, A, D))
    out_rots = template_rot.new_zeros((Ns, V, 3, 3))
    out_stable = stable_list[0].new_zeros((Ns, V, A)) if has_stable else None

    # per object: gather only the rows whose nearest cloud point lies in that object
    start = 0
    for k, pts in enumerate(transformed_pts):
        end = start + pts.size(0)
        # seed points in object k; rows(1,1024), return index of non-zero elements in the mask (1,Np)
        rows = torch.nonzero((seed_idx >= start) & (seed_idx < end), as_tuple=True)[0] 
        if rows.numel() > 0:
            local = seed_idx[rows] - start                  # index inside object k
            vmap = view_maps[k]
            out_scores[rows] = torch.index_select(scores_list[k][local], 1, vmap) # num_obj * (Np_k, V, A, D) -> (Ns_k, V, A, D)
            out_widths[rows] = torch.index_select(widths_list[k][local], 1, vmap) # num_obj * (Np_k, V, A, D) -> (Ns_k, V, A, D)
            out_rots[rows] = view_rots[k]                   # (1024,V,3,3)
            if has_stable:
                out_stable[rows] = torch.index_select(stable_list[k][local], 1, vmap) # num_obj * (Np_k, V, A) -> (Ns_k, V, A)
        start = end

    return all_pts[seed_idx], out_rots, out_scores, out_widths, out_stable


def _compute_view_graspness(scores):
    """scores: (B, Ns, V, A, D) -> min-max normalised view graspness (B, Ns, V)."""
    good = (scores > 0) & (scores <= VIEW_U_THRESHOLD)
    g = good.to(scores.dtype).flatten(start_dim=3).sum(dim=-1) / VIEW_GRASP_NUM
    lo = g.amin(dim=-1, keepdim=True)
    hi = g.amax(dim=-1, keepdim=True)
    return (g - lo) / (hi - lo + 1e-5)


def process_grasp_labels(end_points):
    """ generate labels according to scene points and object poses. """
    seed_xyzs = end_points['xyz_graspable']            # (B, Ns, 3)
    has_stable = 'grasp_stable_list' in end_points
    batch = seed_xyzs.size(0)

    V = end_points['grasp_scores_list'][0][0].size(1)
    template_views = generate_grasp_views(V).to(seed_xyzs.device)
    zero_angle = torch.zeros(V, dtype=template_views.dtype, device=template_views.device)
    # Build template rotation matrices for all views (V, 3, 3) = approach vector + in-plane rotation
    template_rot = batch_viewpoint_params_to_matrix(-template_views, zero_angle)

    samples = [
        _collect_sample(end_points, b, seed_xyzs[b], template_views, template_rot, has_stable)
        for b in range(batch)
    ]
    points, rots, scores, widths, stable = (
        [s[j] for s in samples] for j in range(5)
    )
    points = torch.stack(points) # (B, Ns[1024], 3)
    rots = torch.stack(rots)     # (B, Ns[1024], V, 3, 3)
    scores = torch.stack(scores) # (B, Ns[1024], V, A, D)
    widths = torch.stack(widths) # (B, Ns[1024], V, A, D)

    # view-level graspness uses scores before width filtering, (B, Ns[1024], V)
    view_graspness = _compute_view_graspness(scores) 

    # invalid grasp scores or wider than the gripper 
    scores = torch.where((scores > 0) & (widths <= GRASP_MAX_WIDTH), scores, torch.zeros_like(scores))

    end_points['batch_grasp_score'] = scores
    end_points['batch_grasp_width'] = widths
    end_points['batch_grasp_view_graspness'] = view_graspness
    end_points['batch_grasp_point'] = points
    end_points['batch_grasp_view_rot'] = rots
    if has_stable:
        stable = torch.stack(stable) # (B, Ns[1024], V, A)
        end_points['batch_grasp_stable'] = stable

    return end_points


def match_grasp_view_and_label(end_points):
    """ Select the predicted-view rotations and normalize its grasp scores. """
    top_view_inds = end_points['grasp_top_view_inds']   # (B, Ns)
    view_rots = end_points['batch_grasp_view_rot']      # (B, Ns, V, 3, 3)
    scores = end_points['batch_grasp_score']             # (B, Ns, V, A, D)
    widths = end_points['batch_grasp_width']             # (B, Ns, V, A, D)
    has_stable = 'grasp_stable_list' in end_points

    # rotation of the predicted view for every seed point -> (B, Ns, 3, 3)
    B, Ns = top_view_inds.shape
    select_rots_view_inds = top_view_inds.view(B, Ns, 1, 1, 1).expand(-1, -1, -1, 3, 3)
    top_rots = torch.gather(view_rots, 2, select_rots_view_inds).squeeze(2)
    # 
    B, Ns, _, A, D = scores.size()
    selected_view_inds = top_view_inds.view(B, Ns, 1, 1, 1).expand(-1, -1, 1, A, D) # prepare for gather, dimensions converted (B,Ns) to (B, Ns, 1, A, D)
    end_points['batch_grasp_width'] = torch.gather(widths, 2, selected_view_inds).squeeze(2) # (B, Ns, A, D), dropped view dimension
    end_points['batch_grasp_score'] = torch.gather(scores, 2, selected_view_inds).squeeze(2) # (B, Ns, A, D), dropped view dimension
    
    # transform raw grasp scores(0-1, best-worst) to a normalized score (0-1, worst-best) for training
    view_grasp_scores = end_points['batch_grasp_score']
    positive = view_grasp_scores > 0
    if positive.any():
        u_max = view_grasp_scores.max()
        u_min = view_grasp_scores[positive].min()
        end_points['batch_grasp_score'][positive] = torch.log(u_max / view_grasp_scores[positive]) / (torch.log(u_max / u_min) + 1e-6)

    if has_stable:
        stable = end_points['batch_grasp_stable'] # (B, Ns, V, A)
        selected_stable_view_inds = top_view_inds.view(B, Ns, 1, 1).expand(-1, -1, 1, A)
        end_points['batch_grasp_stable'] = torch.gather(stable, 2, selected_stable_view_inds).squeeze(2)

    
    return top_rots, end_points
