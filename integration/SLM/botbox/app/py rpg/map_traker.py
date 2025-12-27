import math
import os
import random
import sys
import time
from collections import namedtuple

from kornia.feature import DISKFeatures
from loguru import logger


# Настройка логирования
logger.add(
    "map_tracker.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}",
    level="DEBUG",
    rotation="10 MB"
)

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import QThread, Signal, Qt, Slot

from PySide6.QtWidgets import QMainWindow, QHBoxLayout, QWidget, QPushButton, QApplication, QLabel, QVBoxLayout, \
    QTableWidget, QTableWidgetItem, QSlider

from SLM.botbox.Environment import EntityController, Pawn,  Environment, ScreenCapturer
from SLM.groupcontext import group
from SLM.pySide6Ext.pySide6Q import PySide6GlueWidget
from SLM.pySide6Ext.widgets.object_editor import PySide6ObjectEditor, ObjectEditorView

from helper import cv2pixmap, detect_shift

import kornia as K
import kornia.feature as KF

try:
    import torch
    import clip

    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: CLIP or PyTorch not installed. CLIP-based features will be disabled.")

CLIP_AVAILABLE = False

CLIP_LIMIT = 2

CLAHE = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=(8, 8))

Pose2D = namedtuple('Pose2D', ['x', 'y', 'theta'])
Feature = namedtuple('Feature', ['pt', 'descriptor','score', 'map_point_id'], defaults=(None,None))
MapPoint2D = namedtuple('MapPoint2D', ['id', 'x_world', 'y_world', 'descriptor', 'observed_by_kfs'])
KeyFrame = namedtuple('KeyFrame', ['id', 'pose', 'features', 'image_path', 'clip_embedding'], defaults=(None, None))

class FeatureExtractorMatcher:
    def __init__(self, n_features=1000):
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.ratio_thresh = 0.75
        print("Using ORB/BFMatcher for features and matching.")

    def detect_and_compute(self, image_gray):
        cv_kpts, des = self.orb.detectAndCompute(image_gray, None)
        if cv_kpts is None or des is None:
            return []
        features = [Feature(pt=np.array(kp.pt, dtype=np.float32), descriptor=d) for kp, d in zip(cv_kpts, des)]
        return features

    def match_features(self, features1, features2):
        if not features1 or not features2:
            return np.empty((0, 2)), np.empty((0, 2)), [], []

        des1 = np.array([f.descriptor for f in features1])
        des2 = np.array([f.descriptor for f in features2])
        matches_knn = self.bf_matcher.knnMatch(des1, des2, k=2)

        good_matches_cv = []
        for m_arr in matches_knn:
            if len(m_arr) == 2:
                m, n = m_arr
                if m.distance < self.ratio_thresh * n.distance:
                    good_matches_cv.append(m)

        if not good_matches_cv:
            return np.empty((0, 2)), np.empty((0, 2)), [], []

        matched_pts1_list = [features1[m.queryIdx].pt for m in good_matches_cv]
        matched_pts2_list = [features2[m.trainIdx].pt for m in good_matches_cv]
        good_matches_f1_indices = [m.queryIdx for m in good_matches_cv]
        good_matches_f2_indices = [m.trainIdx for m in good_matches_cv]

        return (np.array(matched_pts1_list, dtype=np.float32),
                np.array(matched_pts2_list, dtype=np.float32),
                good_matches_f1_indices,
                good_matches_f2_indices)

class KorniaDiskLightGlueMatcher:
    def __init__(self, disk_max_keypoints: int = 2048, lightglue_flash: bool = True, device: str = None):

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        logger.info(f"KorniaDiskLightGlueMatcher using device: {self.device}")

        self.disk_max_keypoints = disk_max_keypoints

        # Initialize DISK Detector
        try:
            self.detector = KF.DISK.from_pretrained('depth').to(self.device).eval()
            # For more keypoints, you might need to adjust 'n' in detect_and_compute
            # or use KF.DISK(num_features=disk_max_keypoints) and train/load custom weights.
            # The pretrained 'depth' one has its own characteristics.
            logger.info(
                f"Kornia DISK (pretrained 'depth') initialized. Max keypoints during detection: {self.disk_max_keypoints}")
        except Exception as e:
            logger.error(f"Failed to initialize Kornia DISK: {e}")
            raise

        # Initialize LightGlue Matcher for DISK features
        try:
            self.matcher = KF.LightGlue(features='disk', flash=lightglue_flash).to(self.device).eval()
            # Older Kornia versions: KF.LightGlue.from_pretrained('disk', flash=lightglue_flash)
            logger.info(f"Kornia LightGlue (for DISK features, flash_attention={lightglue_flash}) initialized.")
        except Exception as e:
            logger.warning(
                f"Could not initialize LightGlue with flash={lightglue_flash} (Error: {e}). Trying without flash.")
            try:
                self.matcher = KF.LightGlue(features='disk', flash=False).to(self.device).eval()
                logger.info(f"Kornia LightGlue (for DISK features, flash_attention=False) initialized.")
            except Exception as e_no_flash:
                logger.error(f"Failed to initialize Kornia LightGlue even without flash: {e_no_flash}")
                raise

        logger.info("Using Kornia DISK/LightGlue for features and matching.")

    def _preprocess_image_to_tensor(self, image_gray: np.ndarray):
        # Kornia expects BxCxHxW, float32, normalized [0, 1]
        # DISK pretrained model expects 3 channels

        if image_gray.ndim == 2:  # If it's truly grayscale
            # Convert grayscale to 3-channel by repeating the channel
            image_rgb_like = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
        elif image_gray.ndim == 3 and image_gray.shape[2] == 1:  # Grayscale but with a channel dim
            image_rgb_like = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
        elif image_gray.ndim == 3 and image_gray.shape[2] == 3:  # Already BGR/RGB
            image_rgb_like = image_gray  # Assuming it's BGR, K.image_to_tensor handles BGR->RGB if needed
        else:
            raise ValueError(f"Unsupported image shape: {image_gray.shape}")

        # K.image_to_tensor handles BGR to RGB conversion and channel permutation (HWC to CHW)
        img_tensor = K.image_to_tensor(image_rgb_like, keepdim=False).float() / 255.
        img_tensor = img_tensor.to(self.device)  # Shape will be (C, H, W), C=3

        if img_tensor.ndim == 3:  # Ensure batch dimension
            img_tensor = img_tensor.unsqueeze(0)  # Shape: (1, 3, H, W)

        logger.trace(f"Preprocessed image tensor shape for DISK: {img_tensor.shape}")
        return img_tensor

    def detect_and_compute(self, image_gray: np.ndarray):
        if image_gray is None:
            return []

        img_tensor = self._preprocess_image_to_tensor(image_gray)

        with torch.no_grad():
            # DISK output is a list of dicts, one per image in batch
            features_kornia = self.detector(img_tensor, n=self.disk_max_keypoints, pad_if_not_divisible=True)

        futures:DISKFeatures = features_kornia[0]

        if not features_kornia or futures.keypoints.shape[0] == 0:
            return []

        # For batch size 1:
        kps_tensor = futures.keypoints # (N, 2)
        descriptors_tensor = futures.descriptors # (N, D)
        scores_tensor = futures.detection_scores # (N,)

        # Convert to NumPy and list of Feature namedtuples
        kps_np = kps_tensor.cpu().numpy()
        descriptors_np = descriptors_tensor.cpu().numpy()
        scores_np = scores_tensor.cpu().numpy()

        custom_features = [
            Feature(pt=kps_np[i], descriptor=descriptors_np[i], score=scores_np[i])  # map_point_id defaults to None
            for i in range(kps_np.shape[0])
        ]
        return custom_features

    def match_features(self, features1: list, features2: list):
        if not features1 or not features2:
            return np.empty((0, 2), dtype=np.float32), \
                np.empty((0, 2), dtype=np.float32), [], []

        # Prepare data for LightGlue
        # Keypoints: (B, N, 2) tensor, Descriptors: (B, N, D) tensor, Scores: (B, N) tensor
        kpts0 = torch.tensor(np.array([f.pt for f in features1]), device=self.device, dtype=torch.float32).unsqueeze(0)
        desc0 = torch.tensor(np.array([f.descriptor for f in features1]), device=self.device,
                             dtype=torch.float32).unsqueeze(0)
        # Scores are optional for LightGlue but can help if provided
        scores0 = torch.tensor(np.array([f.score for f in features1]), device=self.device,
                               dtype=torch.float32).unsqueeze(0)

        kpts1 = torch.tensor(np.array([f.pt for f in features2]), device=self.device, dtype=torch.float32).unsqueeze(0)
        desc1 = torch.tensor(np.array([f.descriptor for f in features2]), device=self.device,
                             dtype=torch.float32).unsqueeze(0)
        scores1 = torch.tensor(np.array([f.score for f in features2]), device=self.device,
                               dtype=torch.float32).unsqueeze(0)

        # LightGlue expects dictionaries with 'keypoints', 'descriptors', and optionally 'scores'
        # It might also use 'image_size', but for this basic matching, it's often not strictly needed if kpts are pixel coords.
        data_for_lg = {
            'image0': {'keypoints': kpts0, 'descriptors': desc0, 'scores': scores0},
            'image1': {'keypoints': kpts1, 'descriptors': desc1, 'scores': scores1}
        }
        # If you have image sizes, you can add them:
        # data_for_lg['image0']['image_size'] = torch.tensor([[h0, w0]], device=self.device)
        # data_for_lg['image1']['image_size'] = torch.tensor([[h1, w1]], device=self.device)

        with torch.no_grad():
            pred_lg = self.matcher(data_for_lg)

        # pred_lg['matches0'][0] contains indices into kpts1 for each kpt0 (-1 if no match)
        # pred_lg['scores'][0] contains confidence of these matches
        matches0_lg = pred_lg['matches0'][0].cpu().numpy()  # Shape (N0_pts,)
        # match_scores_lg = pred_lg['scores'][0].cpu().numpy() # Confidence scores for matches

        good_matches_f1_indices = np.where(matches0_lg != -1)[0].tolist()
        if not good_matches_f1_indices:
            return np.empty((0, 2), dtype=np.float32), \
                np.empty((0, 2), dtype=np.float32), [], []

        good_matches_f2_indices = matches0_lg[good_matches_f1_indices].tolist()

        matched_pts1_list = np.array([features1[i].pt for i in good_matches_f1_indices], dtype=np.float32)
        matched_pts2_list = np.array([features2[i].pt for i in good_matches_f2_indices], dtype=np.float32)

        return (matched_pts1_list,
                matched_pts2_list,
                good_matches_f1_indices,
                good_matches_f2_indices)


class ImageEmbedderCLIP:
    def __init__(self):
        self.model = None
        self.preprocess = None
        self.device = "cpu"
        if CLIP_AVAILABLE:
            try:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
                print(f"CLIP model ViT-B/32 loaded on {self.device}.")
            except Exception as e:
                print(f"Error loading CLIP model: {e}. CLIP features disabled.")
                self.model = None
        else:
            print("CLIP features disabled (library not found).")

    def get_embedding(self, frame_bgr):
        if not self.model or not self.preprocess:
            return None
        try:
            rgb_image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().numpy().squeeze()
        except Exception as e:
            print(f"Error getting CLIP embedding: {e}")
            return None

class SLAMSystem2DImpl:
    def __init__(self, frame_width, frame_height, feature_config=None,
                 enable_clip=True):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_center = np.array([frame_width / 2.0, frame_height / 2.0], dtype=np.float32)

        _feature_config = feature_config or {}

        kornia_config = {
                    'disk_max_keypoints': _feature_config.get('disk_max_keypoints', 2048),
                    'lightglue_flash': _feature_config.get('lightglue_flash', True)
                }
        self.feature_module = KorniaDiskLightGlueMatcher(**kornia_config)

        self.clip_embedder = None
        if enable_clip and CLIP_AVAILABLE:  # CLIP also needs PyTorch
            self.clip_embedder = ImageEmbedderCLIP()

        self.current_pose = Pose2D(0.0, 0.0, 0.0)
        self.map_points = {}
        self.map_point_id_counter = 0
        self.keyframes = {}
        self.keyframe_id_counter = 0
        self.last_keyframe_id = None
        self.reference_kf_id_for_tracking = None
        self.is_initialized = False
        self.is_lost = False
        self.min_features_for_init = _feature_config.get('min_features_for_init', 20)
        self.min_matches_for_transform = _feature_config.get('min_matches_for_transform',
                                                             8)  # Affine needs at least 3, RANSAC more
        self.ransac_reproj_threshold = _feature_config.get('ransac_reproj_threshold', 5.0)
        self.min_inliers_for_pose = _feature_config.get('min_inliers_for_pose', 6)
        self.kf_min_translation = _feature_config.get('kf_min_translation', 40)
        self.kf_min_rotation = np.deg2rad(_feature_config.get('kf_min_rotation_deg', 8))
        self.kf_min_abs_inliers_for_kf = _feature_config.get('kf_min_abs_inliers_for_kf',
                                                             15)  # Lowered for potentially sparser learned features

        self.map_loc_min_matches = _feature_config.get('map_loc_min_matches', 10)
        self.map_loc_min_inliers = _feature_config.get('map_loc_min_inliers', 8)
        self.loop_closure_min_clip_score = _feature_config.get('loop_closure_min_clip_score', 0.85)
        self.loop_closure_min_geom_inliers = _feature_config.get('loop_closure_min_geom_inliers', 12)
        self.loop_closure_skip_recent_kfs = _feature_config.get('loop_closure_skip_recent_kfs', 10)
    def _preprocess_frame(self, frame_bgr):
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    def _normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _estimate_transform_affine(self, pts1_local, pts2_local_or_global, from_frame_to_global=False):
        if len(pts1_local) < 3:  # Affine needs at least 3 points
            logger.debug(f"Too few points for affine transform: {len(pts1_local)}")
            return None, 0

        # Ensure pts are float32 and correct shape for OpenCV
        pts1 = np.array(pts1_local, dtype=np.float32).reshape(-1, 1, 2)
        pts2 = np.array(pts2_local_or_global, dtype=np.float32).reshape(-1, 1, 2)

        if from_frame_to_global:  # Frame-to-Map (perspective, more general)
            # estimateAffine2D might be too restrictive. Consider estimateRigidTransform or findHomography
            # For 2D SLAM, estimateAffinePartial2D (similarity: rotation, translation, scale) is often better
            # than full affine if shear is not expected.
            # However, your original code used estimateAffine2D for global. Let's stick to it for now.
            # M, inliers_mask = cv2.estimateAffine2D(pts1_centered, pts2_local_or_global, method=cv2.RANSAC,
            #                                        ransacReprojThreshold=self.ransac_reproj_threshold * 1.5)
            # Simpler for now: let's use estimateAffinePartial2D as it's robust for rigid + scale
            # If pts1_local are already centered, no need to subtract frame_center again.
            # The transform should be from image coords to world coords.
            M, inliers_mask = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC,
                                                          ransacReprojThreshold=self.ransac_reproj_threshold * 1.5)  # Increased threshold for global
        else:  # Frame-to-Frame (similarity)
            # pts1_centered = pts1_local - self.frame_center
            # pts2_centered = pts2_local_or_global - self.frame_center
            # M, inliers_mask = cv2.estimateAffinePartial2D(pts1_centered, pts2_centered, method=cv2.RANSAC,
            #                                               ransacReprojThreshold=self.ransac_reproj_threshold)
            M, inliers_mask = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC,
                                                          ransacReprojThreshold=self.ransac_reproj_threshold)

        if M is None:
            logger.debug("estimateAffinePartial2D returned None.")
            return None, 0

        num_inliers = 0
        if inliers_mask is not None:
            num_inliers = np.sum(inliers_mask)
        else:  # Should not happen if M is not None, but as a safeguard
            logger.warning("inliers_mask is None even though M was estimated.")

        logger.debug(f"Affine transform estimated with {num_inliers} inliers out of {len(pts1_local)} matches.")
        return M, num_inliers

    def _transform_points_to_global(self, local_pts_in_frame, pose_of_frame):
        if not isinstance(local_pts_in_frame, np.ndarray):
            local_pts_in_frame = np.array(local_pts_in_frame)
        if local_pts_in_frame.ndim == 1: local_pts_in_frame = local_pts_in_frame.reshape(1, -1)

        # No need to center if local_pts_in_frame are already in image pixel coordinates
        # The transform is: Global = R * (Local - FrameCenter) + PoseTranslation OR
        # Global = R * Local_Centered_At_Origin + PoseTranslation (if Local is already relative to optical center)
        # Assuming local_pts_in_frame are pixel coordinates, and self.frame_center is the optical center.
        local_pts_centered_optical = local_pts_in_frame - self.frame_center

        R = np.array([[math.cos(pose_of_frame.theta), -math.sin(pose_of_frame.theta)],
                      [math.sin(pose_of_frame.theta), math.cos(pose_of_frame.theta)]])

        # Apply rotation then translation
        global_pts = (R @ local_pts_centered_optical.T).T + np.array([pose_of_frame.x, pose_of_frame.y])
        return global_pts

    def _transform_global_to_camera(self, global_pts, pose_of_frame):
        if not isinstance(global_pts, np.ndarray):
            global_pts = np.array(global_pts)
        if global_pts.ndim == 1:
            global_pts = global_pts.reshape(1, -1)

        # Inverse operation:
        # Local_Centered_Optical = R_inv * (Global - PoseTranslation)
        # Local_Pixel = Local_Centered_Optical + FrameCenter

        global_pts_relative_to_pose = global_pts - np.array([pose_of_frame.x, pose_of_frame.y])

        R_inv = np.array([[math.cos(-pose_of_frame.theta), -math.sin(-pose_of_frame.theta)],
                          [math.sin(-pose_of_frame.theta), math.cos(-pose_of_frame.theta)]])

        local_pts_centered_optical = (R_inv @ global_pts_relative_to_pose.T).T
        local_pts_pixel = local_pts_centered_optical + self.frame_center
        return local_pts_pixel

    def initialize(self, frame_bgr):
        frame_gray = self._preprocess_frame(frame_bgr)
        features_initial = self.feature_module.detect_and_compute(frame_gray)

        if len(features_initial) < self.min_features_for_init:
            logger.warning(
                f"Initialization failed: Not enough features ({len(features_initial)} < {self.min_features_for_init}).")
            return False

        initial_pose = Pose2D(0.0, 0.0, 0.0)
        self.current_pose = initial_pose

        kf_id = self.keyframe_id_counter
        self.keyframe_id_counter += 1

        kf_features_with_map_ids = []
        for feat in features_initial:
            mp_id = self.map_point_id_counter
            self.map_point_id_counter += 1

            global_coords = self._transform_points_to_global(feat.pt, initial_pose).squeeze()

            new_mp = MapPoint2D(id=mp_id, x_world=global_coords[0], y_world=global_coords[1],
                                descriptor=feat.descriptor, observed_by_kfs=[kf_id])
            self.map_points[mp_id] = new_mp
            kf_features_with_map_ids.append(feat._replace(map_point_id=mp_id))  # Score is kept from feat

        clip_emb = None
        if self.clip_embedder:
            clip_emb = self.clip_embedder.get_embedding(frame_bgr)

        new_kf = KeyFrame(id=kf_id, pose=initial_pose, features=kf_features_with_map_ids,
                          clip_embedding=clip_emb, image_path=f"kf_{kf_id}.png")  # Store image for debug/future
        self.keyframes[kf_id] = new_kf

        self.last_keyframe_id = kf_id
        self.reference_kf_id_for_tracking = kf_id
        self.is_initialized = True
        self.is_lost = False
        logger.info(f"SLAM Initialized. KF{kf_id} created with {len(kf_features_with_map_ids)} map points.")
        return True

    def _track_motion_core(self, current_features, ref_features_with_ids, ref_pose):
        mkpts_ref_local, mkpts_curr_local, matched_ref_indices, matched_curr_indices = self.feature_module.match_features(
            ref_features_with_ids, current_features)

        if len(mkpts_ref_local) < self.min_matches_for_transform:
            logger.debug(
                f"Track motion: not enough matches ({len(mkpts_ref_local)} < {self.min_matches_for_transform})")
            return None, 0, []

        M_ref_to_curr, num_inliers = self._estimate_transform_affine(mkpts_ref_local, mkpts_curr_local,
                                                                     from_frame_to_global=False)

        if M_ref_to_curr is None or num_inliers < self.min_inliers_for_pose:
            logger.debug(
            f"Track motion: transform estimation failed or too few inliers ({num_inliers} < {self.min_inliers_for_pose})")
            return None, num_inliers, []


        dx_rel = M_ref_to_curr[0, 2]
        dy_rel = M_ref_to_curr[1, 2]


        dtheta_rel = math.atan2(M_ref_to_curr[1, 0], M_ref_to_curr[0, 0])

        new_x = ref_pose.x + dx_rel * math.cos(ref_pose.theta) - dy_rel * math.sin(ref_pose.theta)
        new_y = ref_pose.y + dx_rel * math.sin(ref_pose.theta) + dy_rel * math.cos(ref_pose.theta)


        new_theta = self._normalize_angle(ref_pose.theta + dtheta_rel)

        tracked_curr_features_with_ids = []

        for i, curr_feat_original_idx in enumerate(matched_curr_indices):
            ref_feat_original_idx = matched_ref_indices[i]
            map_point_id = ref_features_with_ids[ref_feat_original_idx].map_point_id
            tracked_curr_features_with_ids.append(
                current_features[curr_feat_original_idx]._replace(map_point_id=map_point_id))

        return Pose2D(new_x, new_y, new_theta), num_inliers, tracked_curr_features_with_ids

    def _localize_globally_core(self, current_features):
        if not self.map_points or not current_features:
            return None, 0, []

        map_features_for_matching = [Feature(pt=None, descriptor=mp.descriptor, map_point_id=mp.id,score=0) for mp in
                                     self.map_points.values()]

        if not map_features_for_matching: return None, 0, []
        mkpts_curr_local, _, matched_curr_indices, matched_map_indices = self.feature_module.match_features(
            current_features, map_features_for_matching)
        if len(mkpts_curr_local) < self.map_loc_min_matches:
            return None, 0, []
        mkpts_map_global = np.array([[self.map_points[map_features_for_matching[idx].map_point_id].x_world,
                                      self.map_points[map_features_for_matching[idx].map_point_id].y_world] for idx in
                                     matched_map_indices], dtype=np.float32)
        M_curr_to_map, num_inliers = self._estimate_transform_affine(mkpts_curr_local, mkpts_map_global,
                                                                     from_frame_to_global=True)
        if M_curr_to_map is None or num_inliers < self.map_loc_min_inliers:
            return None, num_inliers, []
        est_x = M_curr_to_map[0, 2]
        est_y = M_curr_to_map[1, 2]
        est_theta = math.atan2(M_curr_to_map[1, 0], M_curr_to_map[0, 0])
        localized_pose = Pose2D(est_x, est_y, self._normalize_angle(est_theta))
        localized_curr_features_with_ids = []
        for i, curr_feat_original_idx in enumerate(matched_curr_indices):
            map_point_id = map_features_for_matching[matched_map_indices[i]].map_point_id
            localized_curr_features_with_ids.append(
                current_features[curr_feat_original_idx]._replace(map_point_id=map_point_id))
        return localized_pose, num_inliers, localized_curr_features_with_ids

    def _decide_new_keyframe(self, estimated_pose, num_tracked_inliers):
        if self.last_keyframe_id is None: return True
        last_kf = self.keyframes[self.last_keyframe_id]
        translation_diff = np.sqrt((estimated_pose.x - last_kf.pose.x) ** 2 + (estimated_pose.y - last_kf.pose.y) ** 2)
        rotation_diff = abs(self._normalize_angle(estimated_pose.theta - last_kf.pose.theta))
        if num_tracked_inliers < self.kf_min_abs_inliers_for_kf / 2:
            if translation_diff > self.kf_min_translation / 3 or rotation_diff > self.kf_min_rotation / 3:
                return True
        if translation_diff > self.kf_min_translation or rotation_diff > self.kf_min_rotation:
            return True
        return False

    def _add_keyframe_and_update_map(self, frame_bgr, new_pose, features_for_kf_with_potential_ids):
        kf_id = self.keyframe_id_counter
        self.keyframe_id_counter += 1
        clip_emb = None
        if self.clip_embedder:
            clip_emb = self.clip_embedder.get_embedding(frame_bgr)
        updated_kf_features = []
        num_new_map_points = 0
        for feat_in_kf in features_for_kf_with_potential_ids:
            if feat_in_kf.map_point_id is None or feat_in_kf.map_point_id not in self.map_points:
                mp_id = self.map_point_id_counter
                self.map_point_id_counter += 1
                global_coords = self._transform_points_to_global(feat_in_kf.pt, new_pose).squeeze()
                new_mp = MapPoint2D(id=mp_id, x_world=global_coords[0], y_world=global_coords[1],
                                    descriptor=feat_in_kf.descriptor, observed_by_kfs=[kf_id])
                self.map_points[mp_id] = new_mp
                updated_kf_features.append(feat_in_kf._replace(map_point_id=mp_id))
                num_new_map_points += 1
            else:
                self.map_points[feat_in_kf.map_point_id].observed_by_kfs.append(kf_id)
                updated_kf_features.append(feat_in_kf)
        new_kf = KeyFrame(id=kf_id, pose=new_pose, features=updated_kf_features, clip_embedding=clip_emb,
                          image_path=f"kf_{kf_id}.png")
        self.keyframes[kf_id] = new_kf
        self.last_keyframe_id = kf_id
        self.reference_kf_id_for_tracking = kf_id
        print(f"Created KF{kf_id}. Added {num_new_map_points} new map points. Total map points: {len(self.map_points)}")

    def _detect_loop_closure(self, current_kf):
        if not self.clip_embedder or current_kf.clip_embedding is None or len(
                self.keyframes) < self.loop_closure_skip_recent_kfs + 2:
            return None
        candidates = []
        for kf_id, old_kf in self.keyframes.items():
            if kf_id >= current_kf.id - self.loop_closure_skip_recent_kfs: continue
            if old_kf.clip_embedding is None: continue
            score = np.dot(current_kf.clip_embedding, old_kf.clip_embedding) / (
                        np.linalg.norm(current_kf.clip_embedding) * np.linalg.norm(old_kf.clip_embedding))
            if score > self.loop_closure_min_clip_score:
                candidates.append({'id': kf_id, 'score': score, 'kf_obj': old_kf})
        if not candidates: return None
        candidates.sort(key=lambda x: x['score'], reverse=True)
        for cand in candidates[:3]:
            old_kf_obj = cand['kf_obj']
            mkpts_old_local, mkpts_curr_local, _, _ = self.feature_module.match_features(old_kf_obj.features,
                                                                                         current_kf.features)
            if len(mkpts_old_local) < self.loop_closure_min_geom_inliers: continue
            M_loop, num_inliers_loop = self._estimate_transform_affine(mkpts_old_local, mkpts_curr_local,
                                                                       from_frame_to_global=False)
            if M_loop is not None and num_inliers_loop >= self.loop_closure_min_geom_inliers:
                print(
                    f"  LOOP CLOSURE DETECTED between KF{current_kf.id} and KF{old_kf_obj.id}! Score: {cand['score']:.3f}, Inliers: {num_inliers_loop}")
                return old_kf_obj.id, M_loop
        return None

    def _relocalize_with_clip(self, frame_bgr, current_features_all):
        if not self.clip_embedder or not self.keyframes: return None, 0, []
        current_clip_emb = self.clip_embedder.get_embedding(frame_bgr)
        if current_clip_emb is None: return None, 0, []
        best_match_kf_id = -1;
        max_score = -1.0;
        best_kf_obj = None
        for kf_id, kf in self.keyframes.items():
            if kf.clip_embedding is None: continue
            score = np.dot(current_clip_emb, kf.clip_embedding) / (
                        np.linalg.norm(current_clip_emb) * np.linalg.norm(kf.clip_embedding))
            if score > max_score: max_score = score; best_match_kf_id = kf_id; best_kf_obj = kf
        if max_score > self.loop_closure_min_clip_score * 0.9 and best_kf_obj is not None:
            print(
                f"  Relocalization candidate KF{best_match_kf_id} by CLIP (score: {max_score:.3f}). Attempting geometric match...")
            reloc_pose, num_inliers, reloc_features_ids = self._track_motion_core(current_features_all,
                                                                                  best_kf_obj.features,
                                                                                  best_kf_obj.pose)
            if reloc_pose and num_inliers > self.min_inliers_for_pose:
                print(f"    Relocalized successfully to KF{best_match_kf_id}! Inliers: {num_inliers}")
                self.reference_kf_id_for_tracking = best_match_kf_id
                return reloc_pose, num_inliers, reloc_features_ids
        print("  Relocalization with CLIP failed.")
        return None, 0, []

    def process_frame(self, frame_bgr):
        if not self.is_initialized:
            return self.initialize(frame_bgr)
        frame_gray = self._preprocess_frame(frame_bgr)
        current_features_all = self.feature_module.detect_and_compute(frame_gray)
        if not current_features_all:
            print("Lost tracking: No features in current frame.")
            self.is_lost = True
            return False
        estimated_pose = None
        num_inliers_for_kf_decision = 0
        current_features_for_new_kf = list(current_features_all)
        if self.is_lost:
            if self.clip_embedder:
                reloc_pose, num_inliers, reloc_features_ids = self._relocalize_with_clip(frame_bgr,
                                                                                         current_features_all)
                if reloc_pose:
                    estimated_pose = reloc_pose
                    num_inliers_for_kf_decision = num_inliers
                    current_features_for_new_kf = reloc_features_ids
                    self.is_lost = False
            if self.is_lost:
                glob_pose, num_inliers, glob_features_ids = self._localize_globally_core(current_features_all)
                if glob_pose: estimated_pose = glob_pose
                num_inliers_for_kf_decision = num_inliers
                current_features_for_new_kf = glob_features_ids
                self.is_lost = False
                print("  Relocalized by global map search after being lost.")
        else:
            if len(self.map_points) > self.map_loc_min_matches * 2:
                glob_pose, num_inliers, glob_features_ids = self._localize_globally_core(current_features_all)
                if glob_pose: estimated_pose = glob_pose
                num_inliers_for_kf_decision = num_inliers
                current_features_for_new_kf = glob_features_ids
                print( f"  Tracked against map, inliers: {num_inliers_for_kf_decision}")
            if estimated_pose is None and self.reference_kf_id_for_tracking is not None:
                ref_kf = self.keyframes[self.reference_kf_id_for_tracking]
                res = self._track_motion_core(current_features_all,ref_kf.features,ref_kf.pose)
                try:
                    kf_track_pose, num_inliers, kf_track_features_ids = res
                except Exception as e:
                    print(f"Error during tracking against KF{self.reference_kf_id_for_tracking}: {e}")

                if kf_track_pose: estimated_pose = kf_track_pose
                num_inliers_for_kf_decision = num_inliers
                current_features_for_new_kf = kf_track_features_ids
                print(f"  Tracked against KF{self.reference_kf_id_for_tracking}, inliers: {num_inliers_for_kf_decision}")
        if estimated_pose is None:
            print("Lost tracking: Could not estimate pose in this frame.");
            self.is_lost = True;
            return False
        self.current_pose = estimated_pose
        if self._decide_new_keyframe(estimated_pose, num_inliers_for_kf_decision):
            self._add_keyframe_and_update_map(frame_bgr, estimated_pose, current_features_for_new_kf)
            newly_created_kf = self.keyframes[self.last_keyframe_id]
            loop_info = self._detect_loop_closure(newly_created_kf)
            if loop_info: pass  # Future: Use loop_info for graph optimization
        return True




# === SegmentMapper класс ===

class PlayerTracker(EntityController):
    def __init__(self):
        super().__init__()
        self.first_frame = True
        self.camera_position = np.array([0, 0])
        self.player_position = np.array([0, 0])
        self.slam = SLAMSystem2DImpl(1700,500,None,False)

    def update(self):
        # стандартная проверка тайминга
        if not super().update():
            return

        image = self.env.GetControllerByType(ScreenCapturer).grab_buffer["map_tracing_image"]
        if image is None:
            return

        self.slam.process_frame(image)
        self.env.data_board.data["slam_pose"] = self.slam.current_pose



class GameWorldMap(EntityController):
    def __init__(self):
        super().__init__()
        self.name = 'MapTracer'



class Visualizer   (EntityController):
    def __init__(self):
        super().__init__()
        self.name = 'MapTracer'
        mode = "filter_gaussian","tracking"

    def init(self):
        pass

    def render(self):
        image = self.env.GetControllerByType(ScreenCapturer).grab_buffer["full_screen_pil"]
        if image is None:
            return
        image = image.copy()

        player_tracker = self.env.GetControllerByType(PlayerTracker)
        slam_system = player_tracker.slam
        current_pose = slam_system.current_pose

        slam_frame_width = slam_system.frame_width
        slam_frame_height = slam_system.frame_height

        # Draw Map Points
        for mp_id, mp in slam_system.map_points.items():
            mp_global_pt = np.array([mp.x_world, mp.y_world])
            mp_camera_coords_arr = slam_system._transform_global_to_camera(mp_global_pt, current_pose)

            if mp_camera_coords_arr is None or mp_camera_coords_arr.size == 0: continue
            mp_camera_coords = mp_camera_coords_arr.squeeze()

            if 0 <= mp_camera_coords[0] < slam_frame_width and \
                    0 <= mp_camera_coords[1] < slam_frame_height:
                cv2.circle(image,
                           (int(mp_camera_coords[0]), int(mp_camera_coords[1])),
                           3, (0, 255, 0), -1)  # Green dots for map points

        image = cv2.resize(image, (800, 600), interpolation=cv2.INTER_AREA) # need resize to 800*600 specialy for this render
        self.env.renderer.draw_image(image, (0, 0))




class botThread(QThread):
    render_buffer = Signal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.env = Environment()
        self.env.renderer.b_tr = self
        sc_graber = ScreenCapturer()
        sc_graber.grab_region = (0,100,1700,600)

        sc_graber.add_filter(filter_for_tracking)
        self.env.AddController(sc_graber)
        filterssetings = filters_setings()
        self.env.AddController(filterssetings)
        map = GameWorldMap()
        self.env.AddController(map)
        _2d_tracker = PlayerTracker()
        self.env.AddController(_2d_tracker)
        _vizualizer = Visualizer()
        self.env.AddController(_vizualizer)
        player_pawn = Pawn("p_pwn")
        player_pawn.enabled = True
        self.env.AddChild(player_pawn)

    def set_render_buffer(self, buffer):
        self.render_buffer.emit(buffer)

    def run(self):
        self.env.Start()

class filters_setings(EntityController):
    gaus_kernel_size = 5

    def set_gaus_kernel_size(self, size):
        self.gaus_kernel_size = size

def filter_for_tracking(sc_capturer:ScreenCapturer):
    current_frame = sc_capturer.grab_buffer["full_screen_cv"]
    if current_frame is None:
        sc_capturer.grab_buffer["map_tracing_image"] = None
    else:
        _filters_setings = sc_capturer.env.GetControllerByType(filters_setings)
        k_size = _filters_setings.gaus_kernel_size
        blurred = cv2.GaussianBlur(current_frame, (k_size, k_size), 0)
        #enhanced = CLAHE.apply(blurred)
        sc_capturer.grab_buffer["map_tracing_image"] = blurred

class filter_settings_editor(ObjectEditorView):
    def __init__(self,obj):
        super().__init__(obj)
        self.layout = QVBoxLayout(self)
        self.label = QLabel("Filter Settings", self)
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)
        edit_obj:filters_setings = self.object
        slider_min = 1
        slider_max = 10
        slider_value = edit_obj.gaus_kernel_size
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setMinimum(slider_min)
        self.slider.setMaximum(slider_max)
        self.slider.setValue(slider_value)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(1)
        self.slider.valueChanged.connect(self.update_value)

        self.layout.addWidget(self.slider)

    def update_value(self, value):
        edit_obj:filters_setings = self.object
        edit_obj.set_gaus_kernel_size(value)
        #self.object.env.GetControllerByType(Visualizer).render_buffer.emit(self.object.env.GetControllerByType(ScreenCapturer).grab_buffer["map_tracing_image"])




class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.boThread = botThread()
        self.boThread.render_buffer.connect(self.update_image)
        self.setWindowTitle("OpenCV with Qt - Tune Canny Edges")
        self.setGeometry(0, 0, 800, 600)

        container = QWidget()
        self.setCentralWidget(container)
        with group():
            layout = QHBoxLayout()
            container.setLayout(layout)
            self.label = QLabel(self)
            self.label.setAlignment(Qt.AlignCenter)
            with group():
                left_layout = QVBoxLayout()
                layout.addLayout(left_layout)
                self.enableButton = QPushButton('Enable')
                self.enableButton.clicked.connect(self.boThread.start)
                left_layout.addWidget(self.label)
                left_layout.addWidget(self.enableButton)
                add_marker_button = QPushButton('Add marker')
                add_marker_button.clicked.connect(lambda :self.boThread.env.GetControllerByType(GameWorldMap).add_marker())
                left_layout.addWidget(add_marker_button)
                object_editor = PySide6ObjectEditor()
                object_editor.add_view_template(filters_setings,filter_settings_editor)
                object_editor.set_object(self.boThread.env.GetControllerByType(filters_setings))
                left_layout.addWidget(object_editor)
            # Set up the table
            self.table_widget = QTableWidget()
            # on property change
            self.table_widget.setRowCount(5)
            self.table_widget.setColumnCount(2)
            self.table_widget.setHorizontalHeaderLabels(['Parameter', 'Value'])
            layout.addWidget(self.table_widget)


    def update_table(self, data):
        self.table_widget.setRowCount(len(data))
        for i, (key, value) in enumerate(data.items()):
            key_item = QTableWidgetItem(key)
            value_item = QTableWidgetItem(str(value))
            self.table_widget.setItem(i, 0, key_item)
            self.table_widget.setItem(i, 1, value_item)

    @Slot(np.ndarray)
    def update_image(self, cv_img):
        # if cv_image is filed with zeros, do nothing

        # dawn_scale to 800*600
        #cv_img = cv2.resize(cv_img, (800, 600), interpolation=cv2.INTER_AREA)

        q_img = cv2pixmap(cv_img)
        self.label.setPixmap(q_img)
        data_board = self.boThread.env.data_board
        self.update_table(data_board.data)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
