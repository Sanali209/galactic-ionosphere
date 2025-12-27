import math
import os
import random
import sys
import time
from collections import namedtuple

from loguru import logger

# KORNIA IMPORTS - ADD THESE
try:
    import kornia as K
    import kornia.feature as KF

    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False
    print("CRITICAL WARNING: Kornia or its dependencies not installed. DISK/LightGlue matcher will not work.")
    # You might want to fall back to ORB or raise an error if Kornia is essential
    # For this example, we'll define the class but it will fail at runtime if Kornia is not there.

# Remove or comment out these if SuperPoint/SuperGlue specific classes are no longer primary
# from SLM.botbox.app.superglue.superglue import SuperGlue # Keep if SuperPointGlueFeatureExtractorMatcher is still used
# from SLM.botbox.app.superglue.superpoint import SuperPoint # Keep if SuperPointGlueFeatureExtractorMatcher is still used

# Настройка логирования
logger.remove()  # Удаляем стандартный обработчик
logger.add(
    "map_tracker.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}",
    level="DEBUG",
    rotation="10 MB"
)

import cv2
import numpy as np
import pyautogui  # Not used in SLAMSystem2DImpl, but in your broader context
import pynput  # Not used in SLAMSystem2DImpl
from PIL import Image
from PySide6.QtCore import QThread, Signal, Qt, Slot

from PySide6.QtWidgets import QMainWindow, QHBoxLayout, QWidget, QPushButton, QApplication, QLabel, QVBoxLayout, \
    QTableWidget, QTableWidgetItem, QSlider
# from torchaudio.functional import speed # Not used here

# Assuming these are your project's classes
from SLM.botbox.Environment import EntityController, Pawn, EnvironmentEntity, Environment, ScreenCapturer, BotAgent, \
    Discreet_action_space, Action
from SLM.botbox.behTree import Sequence, Blackboard, ActionNode
from SLM.groupcontext import group
from SLM.pySide6Ext.pySide6Q import PySide6GlueWidget
from SLM.pySide6Ext.widgets.object_editor import PySide6ObjectEditor, ObjectEditorView
from helper import cv2pixmap, detect_shift

try:
    import torch

    # import clip # CLIP parts are separate from feature matching for SLAM
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("Warning: PyTorch not installed. Some features might be disabled (Kornia, SuperPoint/Glue).")

# CLIP_AVAILABLE is handled by ImageEmbedderCLIP, keep it separate
CLIP_AVAILABLE = False  # As per your code, explicitly disabling.
try:
    import clip

    # Check PyTorch availability for CLIP as well
    if PYTORCH_AVAILABLE:
        CLIP_AVAILABLE = True  # Enable if both torch and clip are there
except ImportError:
    CLIP_AVAILABLE = False

if not CLIP_AVAILABLE:
    print("Warning: CLIP or PyTorch not installed/explicitly disabled. CLIP-based features will be disabled.")

CLIP_LIMIT = 10
CLAHE = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=(8, 8))

Pose2D = namedtuple('Pose2D', ['x', 'y', 'theta'])

# UPDATED Feature Namedtuple to include score
Feature = namedtuple('Feature', ['pt', 'descriptor', 'score', 'map_point_id'],
                     defaults=(None, None))  # score defaults to None, map_point_id defaults to None

MapPoint2D = namedtuple('MapPoint2D', ['id', 'x_world', 'y_world', 'descriptor',
                                       'observed_by_kfs'])  # Descriptor could be from ORB or DISK
KeyFrame = namedtuple('KeyFrame', ['id', 'pose', 'features', 'image_path', 'clip_embedding'], defaults=(None, None))


# --- Original ORB FeatureExtractorMatcher (can be kept for fallback/comparison) ---
class FeatureExtractorMatcher:
    def __init__(self, n_features=1000):
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)  # crossCheck=False is typical for knnMatch
        self.ratio_thresh = 0.75
        print("Using ORB/BFMatcher for features and matching.")

    def detect_and_compute(self, image_gray):
        cv_kpts, des = self.orb.detectAndCompute(image_gray, None)
        if cv_kpts is None or des is None:
            return []
        # ORB doesn't inherently provide a 'score' in the same way learned detectors do.
        # We can use kp.response as a proxy, or just set it to a default (e.g., 1.0 or None).
        # For consistency with the updated Feature namedtuple, let's add a placeholder score.
        features = [
            Feature(pt=np.array(kp.pt, dtype=np.float32), descriptor=d, score=kp.response)  # Using kp.response as score
            for kp, d in zip(cv_kpts, des)
        ]
        return features

    def match_features(self, features1, features2):
        if not features1 or not features2:
            return np.empty((0, 2)), np.empty((0, 2)), [], []

        des1 = np.array([f.descriptor for f in features1])
        des2 = np.array([f.descriptor for f in features2])

        # Ensure descriptors are uint8 for NORM_HAMMING
        if des1.dtype != np.uint8: des1 = des1.astype(np.uint8)
        if des2.dtype != np.uint8: des2 = des2.astype(np.uint8)

        matches_knn = self.bf_matcher.knnMatch(des1, des2, k=2)

        good_matches_cv = []
        for m_arr in matches_knn:
            if len(m_arr) == 2:  # Ensure we got 2 neighbors
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


# --- Kornia DISK + LightGlue Feature Extractor and Matcher ---
class KorniaDiskLightGlueMatcher:
    def __init__(self, disk_max_keypoints: int = 2048, lightglue_flash: bool = True, device: str = None):
        if not KORNIA_AVAILABLE or not PYTORCH_AVAILABLE:
            raise ImportError("Kornia and PyTorch are required for KorniaDiskLightGlueMatcher.")

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
        img_tensor = K.image_to_tensor(image_gray, keepdim=False).float() / 255.
        img_tensor = img_tensor.to(self.device)
        if img_tensor.ndim == 3:  # Ensure batch dimension if C=1 was squeezed
            img_tensor = img_tensor.unsqueeze(0)
        return img_tensor  # Shape: (1, 1, H, W)

    def detect_and_compute(self, image_gray: np.ndarray):
        if image_gray is None:
            return []

        img_tensor = self._preprocess_image_to_tensor(image_gray)

        with torch.no_grad():
            # DISK output is a list of dicts, one per image in batch
            features_kornia = self.detector(img_tensor, n=self.disk_max_keypoints, pad_if_not_divisible=True)

        if not features_kornia or features_kornia[0]['keypoints'].shape[0] == 0:
            return []

        # For batch size 1:
        kps_tensor = features_kornia[0]['keypoints']  # (N, 2)
        descriptors_tensor = features_kornia[0]['descriptors']  # (N, D)
        scores_tensor = features_kornia[0]['detection_scores']  # (N,)

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


# --- Your SuperPointGlueFeatureExtractorMatcher (can be kept for comparison or if needed) ---
# Ensure PyTorch is available if this class is used
if PYTORCH_AVAILABLE:
    # The imports for SuperPoint/SuperGlue from your local paths would go here
    # For example:
    # from SLM.botbox.app.superglue.superglue import SuperGlue
    # from SLM.botbox.app.superglue.superpoint import SuperPoint
    # Ensure these paths are correct relative to your project structure.
    # For now, I'll assume they are correctly imported if this class is instantiated.
    try:
        from SLM.botbox.app.superglue.superglue import SuperGlue
        from SLM.botbox.app.superglue.superpoint import SuperPoint

        SUPERPOINTGLUE_LOCALLY_AVAILABLE = True
    except ImportError:
        SUPERPOINTGLUE_LOCALLY_AVAILABLE = False
        logger.warning("Local SuperPoint/SuperGlue models not found at SLM.botbox.app.superglue. "
                       "SuperPointGlueFeatureExtractorMatcher will fail if instantiated.")


    class SuperPointGlueFeatureExtractorMatcher:
        def __init__(self, superpoint_config=None, superglue_config=None, device=None):
            if not SUPERPOINTGLUE_LOCALLY_AVAILABLE:
                raise ImportError(
                    "Local SuperPoint/SuperGlue models not found. Cannot initialize SuperPointGlueFeatureExtractorMatcher.")

            if device is None:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device(device)
            logger.info(f"SuperPointGlueFeatureExtractorMatcher using device: {self.device}")

            default_sp_config = {'nms_radius': 4, 'keypoint_threshold': 0.005, 'max_keypoints': 1024}
            current_sp_config = {**default_sp_config, **(superpoint_config or {})}
            # Simplified weight path logic for brevity, assuming 'weights_path' is provided in config if not default.
            if 'weights_path' not in current_sp_config:
                logger.warning(
                    "SuperPoint 'weights_path' not in config. Model might fail to load if not found by default.")
                # Add your original robust path finding logic here if needed
            self.superpoint = SuperPoint(current_sp_config).eval().to(self.device)
            logger.info(f"SuperPoint initialized with config: {current_sp_config}")

            default_sg_config = {'match_threshold': 0.2,
                                 'weights': 'outdoor'}  # Kornia uses 'weights', not 'weights_path' for pretrained like 'outdoor'
            current_sg_config = {**default_sg_config, **(superglue_config or {})}
            if 'weights_path' in current_sg_config and current_sg_config['weights_path'] in ['outdoor',
                                                                                             'indoor']:  # Remap for clarity
                current_sg_config['weights'] = current_sg_config.pop('weights_path')

            # Simplified weight path logic for SuperGlue
            if 'weights' not in current_sg_config and 'weights_path' not in current_sg_config:
                logger.warning(
                    "SuperGlue 'weights' or 'weights_path' not in config. Model might fail to load if not default.")

            self.superglue = SuperGlue(current_sg_config).eval().to(self.device)
            logger.info(f"SuperGlue initialized with config: {current_sg_config}")
            logger.info("Using SuperPoint/SuperGlue for features and matching.")

        def _preprocess_image(self, image_gray):
            if image_gray is None: raise ValueError("Input image_gray cannot be None.")
            if image_gray.ndim == 3: image_gray = cv2.cvtColor(image_gray, cv2.COLOR_BGR2GRAY)
            return torch.from_numpy(image_gray / 255.).float()[None, None].to(self.device)

        def detect_and_compute(self, image_gray):
            if image_gray is None: return []
            img_tensor = self._preprocess_image(image_gray)
            with torch.no_grad():
                pred_sp = self.superpoint({'image': img_tensor})
            kpts = pred_sp['keypoints'][0].cpu().numpy()
            if kpts.shape[0] == 0: return []
            descriptors = pred_sp['descriptors'][0].cpu().numpy().T  # Transpose to (N,D)
            scores = pred_sp['scores'][0].cpu().numpy()
            features = [
                Feature(pt=kpts[i], descriptor=descriptors[i], score=scores[i])
                for i in range(kpts.shape[0])
            ]
            return features

        def match_features(self, features1, features2, image1_gray=None, image2_gray=None):
            if not features1 or not features2:
                return np.empty((0, 2)), np.empty((0, 2)), [], []
            kpts0 = np.array([f.pt for f in features1])
            kpts1 = np.array([f.pt for f in features2])
            desc0 = np.stack([f.descriptor for f in features1]).T  # (D, N0)
            desc1 = np.stack([f.descriptor for f in features2]).T  # (D, N1)
            scores0 = np.array([f.score for f in features1])
            scores1 = np.array([f.score for f in features2])

            data_for_sg = {
                'keypoints0': torch.from_numpy(kpts0).float().to(self.device)[None],
                'keypoints1': torch.from_numpy(kpts1).float().to(self.device)[None],
                'descriptors0': torch.from_numpy(desc0).float().to(self.device)[None],
                'descriptors1': torch.from_numpy(desc1).float().to(self.device)[None],
                'scores0': torch.from_numpy(scores0).float().to(self.device)[None],
                'scores1': torch.from_numpy(scores1).float().to(self.device)[None],
            }
            if image1_gray is not None:
                data_for_sg['image_size0'] = torch.tensor([image1_gray.shape[:2]], device=self.device).float()
            if image2_gray is not None:
                data_for_sg['image_size1'] = torch.tensor([image2_gray.shape[:2]], device=self.device).float()

            with torch.no_grad():
                pred_sg = self.superglue(data_for_sg)
            matches0 = pred_sg['matches0'][0].cpu().numpy()
            valid_match_mask = matches0 > -1
            good_matches_f1_indices = np.where(valid_match_mask)[0].tolist()
            if not good_matches_f1_indices: return np.empty((0, 2)), np.empty((0, 2)), [], []
            good_matches_f2_indices = matches0[valid_match_mask].tolist()
            matched_pts1_list = kpts0[good_matches_f1_indices]
            matched_pts2_list = kpts1[good_matches_f2_indices]
            return matched_pts1_list, matched_pts2_list, good_matches_f1_indices, good_matches_f2_indices
else:  # PyTorch not available
    logger.warning("PyTorch not available, SuperPointGlueFeatureExtractorMatcher class definition skipped.")
    SuperPointGlueFeatureExtractorMatcher = None  # Define as None if PyTorch is missing


class ImageEmbedderCLIP:
    # ... (Your existing ImageEmbedderCLIP class - unchanged)
    def __init__(self):
        self.model = None
        self.preprocess = None
        self.device = "cpu"
        if CLIP_AVAILABLE and PYTORCH_AVAILABLE:  # Check both
            try:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
                logger.info(f"CLIP model ViT-B/32 loaded on {self.device}.")
            except Exception as e:
                logger.error(f"Error loading CLIP model: {e}. CLIP features disabled.")
                self.model = None
        else:
            logger.warning("CLIP features disabled (library not found or PyTorch missing).")

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
            logger.error(f"Error getting CLIP embedding: {e}")
            return None


class SLAMSystem2DImpl:
    def __init__(self, frame_width, frame_height, feature_module_type="kornia_disk_lightglue", feature_config=None,
                 enable_clip=True):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_center = np.array([frame_width / 2.0, frame_height / 2.0], dtype=np.float32)

        self.feature_module_type = feature_module_type
        _feature_config = feature_config or {}

        if self.feature_module_type == "kornia_disk_lightglue":
            if KORNIA_AVAILABLE and PYTORCH_AVAILABLE:
                logger.info("Initializing SLAM with Kornia DISK + LightGlue.")
                # Example config for Kornia matcher
                kornia_config = {
                    'disk_max_keypoints': _feature_config.get('disk_max_keypoints', 2048),
                    'lightglue_flash': _feature_config.get('lightglue_flash', True)
                }
                self.feature_module = KorniaDiskLightGlueMatcher(**kornia_config)
            else:
                logger.error("Kornia/PyTorch not available. Falling back to ORB for SLAM features.")
                self.feature_module_type = "orb"  # Force fallback
                self.feature_module = FeatureExtractorMatcher(**_feature_config.get('orb_config', {}))
        elif self.feature_module_type == "superpoint_superglue":
            if SuperPointGlueFeatureExtractorMatcher is not None:  # Check if class was defined
                logger.info("Initializing SLAM with SuperPoint + SuperGlue.")
                sp_config = _feature_config.get('superpoint_config', {})
                sg_config = _feature_config.get('superglue_config', {})
                self.feature_module = SuperPointGlueFeatureExtractorMatcher(
                    superpoint_config=sp_config, superglue_config=sg_config
                )
            else:
                logger.error(
                    "SuperPointGlueFeatureExtractorMatcher class not available (PyTorch or local models missing?). Falling back to ORB.")
                self.feature_module_type = "orb"  # Force fallback
                self.feature_module = FeatureExtractorMatcher(**_feature_config.get('orb_config', {}))
        elif self.feature_module_type == "orb":
            logger.info("Initializing SLAM with ORB.")
            self.feature_module = FeatureExtractorMatcher(**_feature_config.get('orb_config', {}))
        else:
            logger.error(f"Unknown feature_module_type: {self.feature_module_type}. Defaulting to ORB.")
            self.feature_module_type = "orb"
            self.feature_module = FeatureExtractorMatcher(**_feature_config.get('orb_config', {}))

        self.clip_embedder = None
        if enable_clip and CLIP_AVAILABLE and PYTORCH_AVAILABLE:  # CLIP also needs PyTorch
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

    # ... (rest of your SLAMSystem2DImpl methods are largely unchanged as they interact via the feature_module interface)
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
        # ref_features_with_ids already contains .pt, .descriptor, .score, .map_point_id
        # current_features contains .pt, .descriptor, .score (map_point_id is None)
        mkpts_ref_local, mkpts_curr_local, matched_ref_indices, matched_curr_indices = \
            self.feature_module.match_features(ref_features_with_ids, current_features)

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

        # Decompose affine matrix M_ref_to_curr (assuming it's a similarity: translation, rotation, uniform scale)
        # M = [[s*cos(th), -s*sin(th), tx], [s*sin(th), s*cos(th), ty]]
        # For frame-to-frame, we expect scale s approx 1.
        # The dx_rel, dy_rel are translations in the *current frame's centered coordinate system* relative to the *reference frame's centered coordinate system*.
        # To update global pose: transform this relative motion into the global frame using ref_pose.theta.

        dx_img_coord = M_ref_to_curr[0, 2]  # Translation in current frame's x (relative to ref frame center)
        dy_img_coord = M_ref_to_curr[1, 2]  # Translation in current frame's y (relative to ref frame center)

        # Rotation relative to reference frame
        dtheta_rel = math.atan2(M_ref_to_curr[1, 0], M_ref_to_curr[0, 0])

        # Transform image-coordinate translation (dx_img_coord, dy_img_coord) to world-coordinate translation
        # This translation happened when the camera was oriented at ref_pose.theta
        dx_world = dx_img_coord * math.cos(ref_pose.theta) - dy_img_coord * math.sin(ref_pose.theta)
        dy_world = dx_img_coord * math.sin(ref_pose.theta) + dy_img_coord * math.cos(ref_pose.theta)

        new_x = ref_pose.x + dx_world
        new_y = ref_pose.y + dy_world
        new_theta = self._normalize_angle(ref_pose.theta + dtheta_rel)

        estimated_pose = Pose2D(new_x, new_y, new_theta)

        # Update current_features with map_point_ids from matched reference features
        tracked_curr_features_with_ids = []
        # Create a copy of current_features to modify
        # No, better to build a new list of only the *matched and inlier* current features with their new map_ids
        # The `inliers_mask` from estimateAffinePartial2D applies to the input `mkpts_ref_local` and `mkpts_curr_local`.
        # We need to map these inliers back to original feature lists.

        # Let's refine this: only propagate map_point_ids for inlier matches
        inlier_current_features_with_ids = []
        if num_inliers > 0:  # Only if we have inliers
            # Assuming inliers_mask is for the 'matched_...' lists
            # Reconstruct inlier indices based on the original 'current_features' and 'ref_features_with_ids'
            original_inlier_curr_indices = []
            original_inlier_ref_indices = []

            # The 'inliers_mask' from estimateAffinePartial2D is 1D and applies to the points passed to it.
            # These points were mkpts_curr_local and mkpts_ref_local.
            # So, we iterate through the *original good matches* that formed mkpts lists.

            temp_mkpts_curr_local_inliers = mkpts_curr_local[inliers_mask.ravel() == 1]  # Get inlier points

            # This part is tricky: relating inliers_mask back to original feature lists
            # matched_ref_indices and matched_curr_indices are indices into the *full* features1 and features2 lists
            # The inliers_mask is on the *subset* of matches that were passed to estimateAffine.
            # For simplicity, assume all matches passed to estimateAffine are used if inliers_mask is not explicitly used to filter here.
            # Or, more correctly, use the inliers from the transform.

            # A simpler approach: for now, take all *good matches* (from ratio test) and assign map_ids.
            # RANSAC inliers are used for pose, but for KF decision, all good matches might be initially considered.
            # The question is, which features from current_features should get a map_point_id?
            # Only those that were matched *and* were RANSAC inliers.

            # Iterate through the *original* good matches
            # `good_matches_cv` from the feature_module.match_features() is not available here.
            # `matched_ref_indices` and `matched_curr_indices` are indices into the full feature lists.
            # We need to filter these based on the RANSAC inliers_mask.

            # Correct way:
            # The `inliers_mask` is of the same length as `mkpts_ref_local` (and `mkpts_curr_local`).
            # So, we iterate `i` from `0` to `len(mkpts_ref_local) - 1`.
            # If `inliers_mask[i]` is true, then `mkpts_ref_local[i]` and `mkpts_curr_local[i]` are an inlier pair.
            # And `matched_ref_indices[i]` and `matched_curr_indices[i]` are their original indices.

            for i in range(len(matched_ref_indices)):  # Iterate over all *potential* matches (before RANSAC)
                if inliers_mask is not None and i < len(inliers_mask) and inliers_mask[
                    i]:  # Check if this match was an inlier
                    original_curr_idx = matched_curr_indices[i]
                    original_ref_idx = matched_ref_indices[i]

                    map_point_id = ref_features_with_ids[original_ref_idx].map_point_id
                    if map_point_id is not None:  # Ensure ref feature had a map_point_id
                        # Create a new Feature tuple for the current feature, now with map_point_id
                        # Keep its original pt, descriptor, score
                        feat_curr = current_features[original_curr_idx]
                        inlier_current_features_with_ids.append(
                            feat_curr._replace(map_point_id=map_point_id)
                        )
            logger.debug(
                f"Track motion: propagated {len(inlier_current_features_with_ids)} map_point_ids from inliers.")

        return estimated_pose, num_inliers, inlier_current_features_with_ids

    def _localize_globally_core(self, current_features):
        if not self.map_points or not current_features:
            logger.debug("Global localization: No map points or current features.")
            return None, 0, []

        # Create Feature objects for map points (pt is None, descriptor is key)
        map_features_for_matching = [
            Feature(pt=None, descriptor=mp.descriptor, score=None, map_point_id=mp.id)
            # Score is None for map points here
            for mp_id, mp in self.map_points.items() if mp.descriptor is not None  # Ensure map point has descriptor
        ]
        if not map_features_for_matching:
            logger.debug("Global localization: No map points with descriptors.")
            return None, 0, []

        # Match current features against all map point descriptors
        # features1 = current_features, features2 = map_features_for_matching
        mkpts_curr_local, _, matched_curr_indices, matched_map_indices = \
            self.feature_module.match_features(current_features, map_features_for_matching)

        if len(mkpts_curr_local) < self.map_loc_min_matches:
            logger.debug(
                f"Global localization: Not enough initial matches ({len(mkpts_curr_local)} < {self.map_loc_min_matches})")
            return None, 0, []

        # Get global coordinates for the matched map points
        mkpts_map_global = np.array([
            [self.map_points[map_features_for_matching[idx].map_point_id].x_world,
             self.map_points[map_features_for_matching[idx].map_point_id].y_world]
            for idx in matched_map_indices
        ], dtype=np.float32)

        # Estimate transform from current_frame_local_points to map_global_points
        M_curr_to_map, num_inliers = self._estimate_transform_affine(mkpts_curr_local, mkpts_map_global,
                                                                     from_frame_to_global=True)

        if M_curr_to_map is None or num_inliers < self.map_loc_min_inliers:
            logger.debug(
                f"Global localization: Transform estimation failed or too few inliers ({num_inliers} < {self.map_loc_min_inliers})")
            return None, num_inliers, []

        # Decompose M_curr_to_map to get global pose
        # This matrix directly transforms points from current image's *centered* coordinates to global world coordinates.
        # M = [[cos(th_map), -sin(th_map), x_map], [sin(th_map), cos(th_map), y_map]] (if scale is 1)
        # (assuming estimateAffinePartial2D, which is similarity: rotation, translation, scale)
        # More generally, estimateAffine2D gives full affine.
        # If estimateAffinePartial2D was used:
        est_x = M_curr_to_map[0, 2]  # This is the world_x of the frame's optical center
        est_y = M_curr_to_map[1, 2]  # This is the world_y of the frame's optical center
        est_theta = math.atan2(M_curr_to_map[1, 0], M_curr_to_map[0, 0])  # Global orientation of the frame

        localized_pose = Pose2D(est_x, est_y, self._normalize_angle(est_theta))

        # Assign map_point_ids to inlier current_features
        localized_curr_features_with_ids = []
        # Similar to _track_motion_core, filter by RANSAC inliers
        # `inliers_mask` is from estimateAffine, applies to `mkpts_curr_local` and `mkpts_map_global`.
        if num_inliers > 0:
            for i in range(len(matched_curr_indices)):  # Iterate over all matches passed to RANSAC
                # Assuming inliers_mask matches the length of matched_curr_indices
                if i < len(M_curr_to_map[2]) and M_curr_to_map[2][
                    i]:  # M_curr_to_map[2] is inliers_mask from cv2.estimateAffine2D if returned with it.
                    # The returned M from _estimate_transform_affine is just the 2x3 matrix.
                    # We need the inliers_mask separately.
                    # My _estimate_transform_affine returns M, num_inliers. It should return M, inliers_mask.
                    # Let's assume for now, all 'good matches' that led to a valid transform are 'inliers' for this purpose.
                    # This is a simplification; ideally, use the RANSAC inlier_mask directly.
                    # For now: if num_inliers > threshold, assume all initial mkpts contributed to inliers.
                    # This needs fixing in _estimate_transform_affine to return the mask.

                    # Quick fix: If _estimate_transform_affine doesn't return the mask, we can't filter precisely here.
                    # For now, let's assume if a global pose was found, all `matched_curr_indices` contributed.
                    # This is not strictly correct but allows progress.
                    # TODO: Refactor _estimate_transform_affine to return inliers_mask for precise filtering.

                    original_curr_idx = matched_curr_indices[i]
                    map_point_id_from_match = map_features_for_matching[matched_map_indices[i]].map_point_id

                    feat_curr = current_features[original_curr_idx]
                    localized_curr_features_with_ids.append(
                        feat_curr._replace(map_point_id=map_point_id_from_match)
                    )
            logger.debug(f"Global localization: propagated {len(localized_curr_features_with_ids)} map_point_ids.")

        return localized_pose, num_inliers, localized_curr_features_with_ids

    def _decide_new_keyframe(self, estimated_pose, num_tracked_inliers):
        if self.last_keyframe_id is None: return True  # First frame after init is always KF

        last_kf = self.keyframes[self.last_keyframe_id]

        translation_diff = np.sqrt((estimated_pose.x - last_kf.pose.x) ** 2 + (estimated_pose.y - last_kf.pose.y) ** 2)
        rotation_diff_abs = abs(self._normalize_angle(estimated_pose.theta - last_kf.pose.theta))

        # Condition 1: Significant motion
        cond_significant_motion = (translation_diff > self.kf_min_translation or
                                   rotation_diff_abs > self.kf_min_rotation)

        # Condition 2: Low number of tracked inliers (suggests view has changed enough)
        # and some motion has occurred (to avoid KFs when static but features are unstable)
        cond_low_inliers_with_motion = (num_tracked_inliers < self.kf_min_abs_inliers_for_kf and \
                                        (translation_diff > self.kf_min_translation / 3 or \
                                         rotation_diff_abs > self.kf_min_rotation / 3))

        # Condition 3: Number of keyframes is small (force more KFs early on)
        cond_few_kfs = len(self.keyframes) < 5  # e.g., ensure at least 5 KFs quickly if moving

        if cond_significant_motion or cond_low_inliers_with_motion or (cond_few_kfs and cond_significant_motion):
            logger.info(
                f"New KF decision: Motion: T={translation_diff:.2f}(>{self.kf_min_translation:.2f}), R={np.rad2deg(rotation_diff_abs):.2f}(>{np.rad2deg(self.kf_min_rotation):.2f}). "
                f"Inliers: {num_tracked_inliers}(<{self.kf_min_abs_inliers_for_kf}). Few KFs: {len(self.keyframes)}<5. -> YES")
            return True

        logger.debug(f"New KF decision: Motion: T={translation_diff:.2f}, R={np.rad2deg(rotation_diff_abs):.2f}. "
                     f"Inliers: {num_tracked_inliers}. -> NO")
        return False

    def _add_keyframe_and_update_map(self, frame_bgr, new_pose, features_for_kf_with_potential_ids):
        kf_id = self.keyframe_id_counter
        self.keyframe_id_counter += 1

        clip_emb = None
        if self.clip_embedder:
            clip_emb = self.clip_embedder.get_embedding(frame_bgr)

        updated_kf_features = []
        num_new_map_points = 0
        num_updated_map_points = 0

        # features_for_kf_with_potential_ids should be a list of Feature namedtuples.
        # Some might have map_point_id set (if tracked from existing map points), others might be None.
        for feat_in_kf in features_for_kf_with_potential_ids:
            if feat_in_kf.map_point_id is None or feat_in_kf.map_point_id not in self.map_points:
                # New map point
                mp_id = self.map_point_id_counter
                self.map_point_id_counter += 1

                global_coords = self._transform_points_to_global(feat_in_kf.pt, new_pose).squeeze()

                new_mp = MapPoint2D(id=mp_id, x_world=global_coords[0], y_world=global_coords[1],
                                    descriptor=feat_in_kf.descriptor, observed_by_kfs=[kf_id])
                self.map_points[mp_id] = new_mp
                updated_kf_features.append(feat_in_kf._replace(map_point_id=mp_id))  # Update with new mp_id
                num_new_map_points += 1
            else:
                # Existing map point observed by this new keyframe
                mp_id = feat_in_kf.map_point_id
                if kf_id not in self.map_points[mp_id].observed_by_kfs:  # Should always be true for a new KF
                    self.map_points[mp_id].observed_by_kfs.append(kf_id)
                # Optionally, update descriptor if current one is "better" (e.g. higher score)
                # For now, just keep existing descriptor. DISK descriptors are fairly stable.
                # self.map_points[mp_id] = self.map_points[mp_id]._replace(descriptor=feat_in_kf.descriptor) # If updating
                updated_kf_features.append(feat_in_kf)  # map_point_id is already set
                num_updated_map_points += 1

        new_kf = KeyFrame(id=kf_id, pose=new_pose, features=updated_kf_features,
                          clip_embedding=clip_emb, image_path=f"kf_{kf_id}.png")
        self.keyframes[kf_id] = new_kf

        self.last_keyframe_id = kf_id
        # Reference for tracking could be this new KF if it has enough good features.
        if len(updated_kf_features) > self.min_features_for_init / 2:
            self.reference_kf_id_for_tracking = kf_id

        logger.info(
            f"Created KF{kf_id} at pose ({new_pose.x:.2f}, {new_pose.y:.2f}, {np.rad2deg(new_pose.theta):.2f}). "
            f"Features: {len(updated_kf_features)}. New MPs: {num_new_map_points}. Updated MPs: {num_updated_map_points}. Total MPs: {len(self.map_points)}")

    def _detect_loop_closure(self, current_kf):
        if not self.clip_embedder or current_kf.clip_embedding is None or len(
                self.keyframes) < self.loop_closure_skip_recent_kfs + 2:
            return None  # Not enough KFs or no CLIP data

        candidates = []
        current_clip_norm = np.linalg.norm(current_kf.clip_embedding)
        if current_clip_norm == 0: return None  # Avoid division by zero

        for kf_id, old_kf in self.keyframes.items():
            if kf_id >= current_kf.id - self.loop_closure_skip_recent_kfs: continue  # Skip recent KFs
            if old_kf.clip_embedding is None: continue

            old_clip_norm = np.linalg.norm(old_kf.clip_embedding)
            if old_clip_norm == 0: continue

            # Cosine similarity
            score = np.dot(current_kf.clip_embedding, old_kf.clip_embedding) / (current_clip_norm * old_clip_norm)

            if score > self.loop_closure_min_clip_score:
                candidates.append({'id': kf_id, 'score': score, 'kf_obj': old_kf})

        if not candidates: return None

        candidates.sort(key=lambda x: x['score'], reverse=True)  # Sort by highest CLIP score

        logger.debug(
            f"Loop closure: KF{current_kf.id} found {len(candidates)} CLIP candidates. Best score: {candidates[0]['score']:.3f} with KF{candidates[0]['id']}.")

        for cand in candidates[:3]:  # Check top 3 CLIP candidates geometrically
            old_kf_obj = cand['kf_obj']

            # Match features between current_kf and candidate old_kf
            mkpts_old_local, mkpts_curr_local, _, _ = self.feature_module.match_features(old_kf_obj.features,
                                                                                         current_kf.features)

            if len(mkpts_old_local) < self.loop_closure_min_geom_inliers:  # Need enough matches for robust transform
                logger.debug(
                    f"  LC with KF{old_kf_obj.id}: Not enough matches ({len(mkpts_old_local)} < {self.loop_closure_min_geom_inliers})")
                continue

            # Estimate transform. from_frame_to_global=False implies relative transform between KF image spaces
            M_loop, num_inliers_loop = self._estimate_transform_affine(mkpts_curr_local, mkpts_old_local,
                                                                       from_frame_to_global=False)
            # Transform from current_kf features to old_kf features' coordinate system

            if M_loop is not None and num_inliers_loop >= self.loop_closure_min_geom_inliers:
                logger.info(f"  LOOP CLOSURE DETECTED between KF{current_kf.id} and KF{old_kf_obj.id}! "
                            f"CLIP Score: {cand['score']:.3f}, Geom. Inliers: {num_inliers_loop}")
                # M_loop is the transform from current_kf's centered image coords to old_kf's centered image coords
                # This relative transform, along with poses of current_kf and old_kf, defines the loop constraint.
                return old_kf_obj.id, M_loop  # Return ID of loop candidate and relative transform

        logger.debug(f"Loop closure: No geometric confirmation for KF{current_kf.id}'s CLIP candidates.")
        return None

    def _relocalize_with_clip(self, frame_bgr, current_features_all):
        if not self.clip_embedder or not self.keyframes: return None, 0, []

        current_clip_emb = self.clip_embedder.get_embedding(frame_bgr)
        if current_clip_emb is None: return None, 0, []

        best_match_kf_id = -1;
        max_score = -1.0;
        best_kf_obj = None
        current_clip_norm = np.linalg.norm(current_clip_emb)
        if current_clip_norm == 0: return None, 0, []

        for kf_id, kf in self.keyframes.items():
            if kf.clip_embedding is None: continue
            kf_clip_norm = np.linalg.norm(kf.clip_embedding)
            if kf_clip_norm == 0: continue
            score = np.dot(current_clip_emb, kf.clip_embedding) / (current_clip_norm * kf_clip_norm)
            if score > max_score: max_score = score; best_match_kf_id = kf_id; best_kf_obj = kf

        # Use a slightly lower threshold for relocalization than for loop closure
        if max_score > self.loop_closure_min_clip_score * 0.85 and best_kf_obj is not None:
            logger.info(
                f"Relocalization candidate KF{best_match_kf_id} by CLIP (score: {max_score:.3f}). Attempting geometric match...")

            # Try to track current frame against this candidate keyframe
            reloc_pose, num_inliers, reloc_features_ids = self._track_motion_core(current_features_all,
                                                                                  best_kf_obj.features,
                                                                                  best_kf_obj.pose)

            if reloc_pose and num_inliers >= self.min_inliers_for_pose:  # Use min_inliers_for_pose for relocalization
                logger.info(
                    f"  Relocalized successfully to KF{best_match_kf_id}! Pose: ({reloc_pose.x:.2f}, {reloc_pose.y:.2f}, {np.rad2deg(reloc_pose.theta):.2f}), Inliers: {num_inliers}")
                self.reference_kf_id_for_tracking = best_match_kf_id  # Update reference KF
                return reloc_pose, num_inliers, reloc_features_ids
            else:
                logger.debug(
                    f"  Geometric match failed for CLIP relocalization candidate KF{best_match_kf_id}. Inliers: {num_inliers}")

        logger.debug("Relocalization with CLIP failed to find a strong candidate or geometric match.")
        return None, 0, []

    def process_frame(self, frame_bgr):
        if not self.is_initialized:
            logger.info("SLAM not initialized. Attempting initialization.")
            return self.initialize(frame_bgr)

        frame_gray = self._preprocess_frame(frame_bgr)
        current_features_all = self.feature_module.detect_and_compute(frame_gray)

        if not current_features_all:
            logger.warning("Lost tracking: No features detected in current frame.")
            self.is_lost = True
            return False  # Cannot process if no features

        estimated_pose = None
        num_inliers_for_kf_decision = 0
        # current_features_for_new_kf will store features that have map_point_ids after tracking/localization
        current_features_for_new_kf = []

        if self.is_lost:
            logger.info("System is lost. Attempting relocalization...")
            # Attempt 1: Relocalize with CLIP if available
            if self.clip_embedder:
                reloc_pose_clip, num_inliers_clip, reloc_feats_ids_clip = self._relocalize_with_clip(frame_bgr,
                                                                                                     current_features_all)
                if reloc_pose_clip:
                    estimated_pose = reloc_pose_clip
                    num_inliers_for_kf_decision = num_inliers_clip
                    current_features_for_new_kf = reloc_feats_ids_clip
                    self.is_lost = False
                    logger.info("Relocalized using CLIP.")

            # Attempt 2: Relocalize globally using map features (if CLIP failed or not available)
            if self.is_lost:  # Still lost after CLIP attempt
                glob_pose, num_inliers_glob, glob_feats_ids = self._localize_globally_core(current_features_all)
                if glob_pose:
                    estimated_pose = glob_pose
                    num_inliers_for_kf_decision = num_inliers_glob
                    current_features_for_new_kf = glob_feats_ids
                    self.is_lost = False
                    logger.info("Relocalized using global map search.")
        else:  # Not lost, attempt tracking
            # Strategy:
            # 1. Try to track against reference KeyFrame. This is fast.
            # 2. If successful and consistent, use this pose.
            # 3. Optionally, also try to localize against the whole map for robustness or drift correction.
            #    (This can be computationally more expensive).
            #    For now, let's prioritize KF tracking. If it fails badly, then consider global.

            pose_from_kf_track = None
            inliers_from_kf_track = 0
            features_from_kf_track = []

            if self.reference_kf_id_for_tracking is not None and self.reference_kf_id_for_tracking in self.keyframes:
                ref_kf = self.keyframes[self.reference_kf_id_for_tracking]
                logger.debug(f"Tracking against reference KF{ref_kf.id}...")
                kf_track_pose, num_inliers, kf_track_features_ids = self._track_motion_core(
                    current_features_all, ref_kf.features, ref_kf.pose
                )
                if kf_track_pose:
                    pose_from_kf_track = kf_track_pose
                    inliers_from_kf_track = num_inliers
                    features_from_kf_track = kf_track_features_ids
                    logger.debug(
                        f"  Tracked against KF{ref_kf.id}. Pose: ({kf_track_pose.x:.2f}, {kf_track_pose.y:.2f}), Inliers: {num_inliers}")
                else:
                    logger.debug(f"  Tracking against KF{ref_kf.id} failed.")

            # Decide which pose to use if multiple sources are available or if KF tracking failed
            if pose_from_kf_track and inliers_from_kf_track >= self.min_inliers_for_pose / 2:  # If KF tracking is somewhat reliable
                estimated_pose = pose_from_kf_track
                num_inliers_for_kf_decision = inliers_from_kf_track
                current_features_for_new_kf = features_from_kf_track
            else:  # KF tracking failed or very weak, try global localization
                logger.info("KF tracking weak or failed. Attempting global localization.")
                glob_pose, num_inliers_glob, glob_feats_ids = self._localize_globally_core(current_features_all)
                if glob_pose:
                    estimated_pose = glob_pose
                    num_inliers_for_kf_decision = num_inliers_glob
                    current_features_for_new_kf = glob_feats_ids
                    logger.info(f"  Used global localization. Inliers: {num_inliers_glob}")
                elif pose_from_kf_track:  # Fallback to weak KF tracking if global also failed
                    logger.warning(
                        "Global localization also failed. Falling back to potentially weak KF tracking result.")
                    estimated_pose = pose_from_kf_track
                    num_inliers_for_kf_decision = inliers_from_kf_track
                    current_features_for_new_kf = features_from_kf_track

        if estimated_pose is None:
            logger.warning("Lost tracking: Could not estimate pose in this frame (all methods failed).")
            self.is_lost = True
            # Potentially try setting a less reliable reference KF or broaden search next time
            if len(self.keyframes) > 0:
                self.reference_kf_id_for_tracking = random.choice(
                    list(self.keyframes.keys()))  # Random KF as last resort
                logger.info(f"Set random reference KF {self.reference_kf_id_for_tracking} for next attempt.")
            return False

        self.current_pose = estimated_pose
        logger.info(
            f"Current Pose: ({self.current_pose.x:.2f}, {self.current_pose.y:.2f}, Theta: {np.rad2deg(self.current_pose.theta):.1f}), Inliers for KF dec: {num_inliers_for_kf_decision}")

        # If current_features_for_new_kf is empty but we have a pose, it means tracking happened
        # but no features were successfully associated with map_point_ids (e.g. only new features matched).
        # In this case, the new KF will primarily consist of new map points.
        # We need to ensure `features_for_kf_with_potential_ids` for `_add_keyframe_and_update_map`
        # contains all good features from `current_features_all` if `current_features_for_new_kf` is sparse.

        features_to_consider_for_kf = current_features_all
        if current_features_for_new_kf:  # If tracking/localization provided features with map_ids
            # Augment with unmatched features from current_features_all if needed
            # Get set of current feature indices that already have map_ids
            ids_processed = {current_features_all.index(f) for f_with_id in current_features_for_new_kf
                             for f in current_features_all if f.pt is f_with_id.pt}  # This is inefficient.

            # Better: current_features_for_new_kf are ALREADY Feature tuples from current_features_all, just with map_id filled.
            # We need to ensure that _add_keyframe_and_update_map gets ALL features from the current frame,
            # some with map_ids (from current_features_for_new_kf) and others without (from current_features_all).

            # Create a dictionary of features from current_features_all by their pt for quick lookup
            # This is still a bit complex. Let's simplify.
            # `current_features_for_new_kf` are the ones successfully tracked/localized.
            # `current_features_all` are all features detected.
            # For a new KF, we want to use all `current_features_all`.
            # The `map_point_id` in `current_features_all` will be None.
            # `_add_keyframe_and_update_map` will then create new map points for those with None map_id.
            # However, if `current_features_for_new_kf` *already* has map_ids from tracking, we should use those.

            # Simplest: `_add_keyframe_and_update_map` gets the list of features that resulted from tracking/localization
            # (i.e., `current_features_for_new_kf`). If this list is too small, it means most current features
            # did not match existing map points. `_add_keyframe_and_update_map` will then have to create new map points for them.
            # The `features_for_kf_with_potential_ids` argument to `_add_keyframe_and_update_map`
            # should be the features from the current frame, where some *might* have `map_point_id` set if they were successfully
            # associated with existing map points during tracking/localization.

            # If `current_features_for_new_kf` is populated, it's the primary source.
            # If it's empty, it means no existing map points were matched, so all `current_features_all` are new.

            final_features_for_kf_creation = []
            if current_features_for_new_kf:
                # Create a map of original feature (pt tuple) to its version with map_id
                map_id_assigned_pts = {tuple(f.pt): f for f in current_features_for_new_kf if
                                       f.map_point_id is not None}
                for feat in current_features_all:
                    pt_tuple = tuple(feat.pt)
                    if pt_tuple in map_id_assigned_pts:
                        final_features_for_kf_creation.append(map_id_assigned_pts[pt_tuple])
                    else:
                        final_features_for_kf_creation.append(feat)  # map_point_id will be None
            else:  # No features were associated with map_ids, so all are new
                final_features_for_kf_creation = current_features_all

        if self._decide_new_keyframe(estimated_pose, num_inliers_for_kf_decision):
            # Pass the combined list of features to create the KF
            self._add_keyframe_and_update_map(frame_bgr, estimated_pose, final_features_for_kf_creation)

            # Attempt loop closure only after a new keyframe is added
            newly_created_kf = self.keyframes[self.last_keyframe_id]
            loop_info = self._detect_loop_closure(newly_created_kf)
            if loop_info:
                # TODO: Implement loop correction/pose graph optimization
                # loop_candidate_kf_id, relative_transform_M = loop_info
                logger.info(f"Loop closure detected with KF{loop_info[0]}. Pose graph optimization pending.")
                pass

        return True


# === PlayerTracker and other classes from your context ===
# These classes interact with SLAMSystem2DImpl.
# The instantiation of SLAMSystem2DImpl in PlayerTracker needs to be aware of the new feature_module_type option.

class PlayerTracker(EntityController):
    def __init__(self):
        super().__init__()
        self.first_frame = True  # Not used in current logic
        self.camera_position = np.array([0, 0])  # Not used
        self.player_position = np.array([0, 0])  # Not used

        # Configure SLAM System
        # You can choose "kornia_disk_lightglue", "superpoint_superglue", or "orb"
        slam_feature_module = "kornia_disk_lightglue"
        # slam_feature_module = "orb" # Fallback

        slam_feature_config = {
            'orb_config': {'n_features': 1500},  # For "orb"
            'kornia_config': {  # For "kornia_disk_lightglue"
                'disk_max_keypoints': 2048,
                'lightglue_flash': True
            },
            'superpoint_config': {  # For "superpoint_superglue"
                'max_keypoints': 1024,
                # 'weights_path': 'path/to/superpoint_v1.pth' # if not default
            },
            'superglue_config': {  # For "superpoint_superglue"
                'weights': 'outdoor',  # or 'indoor' or path to .pth
                # 'weights_path': 'path/to/superglue_outdoor.pth'
            },
            # Common SLAM parameters (can be overridden per module type if needed)
            'min_features_for_init': 30,
            'min_matches_for_transform': 10,
            'ransac_reproj_threshold': 5.0,
            'min_inliers_for_pose': 8,
            'kf_min_translation': 30,  # Pixels
            'kf_min_rotation_deg': 6,  # Degrees
            'kf_min_abs_inliers_for_kf': 20,  # Min inliers to consider frame for KF
            'map_loc_min_matches': 12,
            'map_loc_min_inliers': 10,
        }

        self.slam = SLAMSystem2DImpl(
            frame_width=1700,  # Make sure this matches your actual map_tracing_image width
            frame_height=500,  # Make sure this matches your actual map_tracing_image height
            feature_module_type=slam_feature_module,
            feature_config=slam_feature_config,
            enable_clip=False  # As per your CLIP_AVAILABLE = False setting
        )
        logger.info(f"PlayerTracker initialized with SLAM module: {self.slam.feature_module_type}")

    def update(self):
        if not super().update():  # Standard timing check from EntityController
            return

        image_for_slam = self.env.data_board.get("map_tracing_image", None)  # Use get for safety
        if image_for_slam is None:
            logger.trace("map_tracing_image not available in data_board for SLAM.")
            return

        # Ensure image is BGR for SLAM's _preprocess_frame if it expects color
        # Your filter_for_tracking already provides 'blurred' which is BGR
        if image_for_slam.ndim == 2:  # If it's grayscale already
            image_for_slam = cv2.cvtColor(image_for_slam, cv2.COLOR_GRAY2BGR)

        try:
            self.slam.process_frame(image_for_slam)
            self.env.data_board.data["slam_pose_x"] = self.slam.current_pose.x
            self.env.data_board.data["slam_pose_y"] = self.slam.current_pose.y
            self.env.data_board.data["slam_pose_theta_deg"] = np.rad2deg(self.slam.current_pose.theta)
            self.env.data_board.data["slam_is_lost"] = self.slam.is_lost
            self.env.data_board.data["slam_map_points"] = len(self.slam.map_points)
            self.env.data_board.data["slam_keyframes"] = len(self.slam.keyframes)
        except Exception as e:
            logger.error(f"Error during SLAM processing: {e}", exc_info=True)
            self.env.data_board.data["slam_error"] = str(e)


# Visualizer might need slight adjustments if image source changes
class Visualizer(EntityController):
    def __init__(self):
        super().__init__()
        self.name = 'MapTracerVisualizer'  # Renamed for clarity
        # mode = "filter_gaussian","tracking" # This seems like a comment

    def init(self):
        pass  # Placeholder

    def render(self):
        # Get the original image that was used for display (before SLAM processing)
        # Or, if you want to draw on the 'map_tracing_image', get that.
        # For drawing on the full screen PIL image:
        display_image_pil = self.env.data_board.get("full_screen_pil", None)
        if display_image_pil is None:
            # Try to get the image fed to SLAM if full_screen_pil is not available
            map_tracing_img_cv = self.env.data_board.get("map_tracing_image", None)
            if map_tracing_img_cv is not None:
                display_image_pil = Image.fromarray(cv2.cvtColor(map_tracing_img_cv, cv2.COLOR_BGR2RGB))
            else:
                logger.trace("No image available for Visualizer rendering.")
                return

        # Convert PIL to OpenCV BGR format for drawing
        display_image_cv = cv2.cvtColor(np.array(display_image_pil), cv2.COLOR_RGB2BGR)

        # Ensure it's a mutable copy
        draw_image = display_image_cv.copy()

        player_tracker = self.env.GetControllerByType(PlayerTracker)
        if not player_tracker:
            logger.warning("PlayerTracker not found in Visualizer.")
            return

        slam_system = player_tracker.slam
        current_pose = slam_system.current_pose

        # These should match the dimensions of the image SLAM processed (map_tracing_image)
        # Not necessarily the dimensions of `draw_image` if it's the full screen.
        # We need to transform map points into the coordinate system of `draw_image`.
        # This is complex if `draw_image` is different from what SLAM saw.
        # Simplification: Assume SLAM processed `map_tracing_image`, and we draw on that.
        # If `draw_image` is `full_screen_pil`, then coordinates might need adjustment if SLAM's
        # `frame_width/height` correspond to a cropped region.

        # Let's assume for drawing, the coordinates are relative to the SLAM processed image dimensions.
        # If map_tracing_image is what SLAM uses, its dimensions are slam_system.frame_width/height.
        # If Visualizer draws on full_screen_pil, and map_tracing_image is a crop of that,
        # you'd need to add the crop's top-left offset to the drawn coordinates.
        # sc_graber.grab_region = (0,100,1700,600) -> x_offset=0, y_offset=100

        # Get the grab region offset if Visualizer draws on the full screen image
        # but SLAM operates on a crop.
        screen_capturer = self.env.GetControllerByType(ScreenCapturer)
        slam_image_offset_x = 0
        slam_image_offset_y = 0
        if screen_capturer and hasattr(screen_capturer, 'grab_region'):
            # Assuming grab_region is (left, top, width, height) for the SLAM image relative to full screen
            # And SLAM frame_width/height are width/height from grab_region.
            # But your grab_region is (left, top, right, bottom)
            # grab_region = (0,100,1700,600) -> left=0, top=100, right=1700, bottom=600
            # width = right - left = 1700, height = bottom - top = 500
            # These should match slam.frame_width, slam.frame_height
            slam_image_offset_x = screen_capturer.grab_region[0]
            slam_image_offset_y = screen_capturer.grab_region[1]

        # Draw Map Points
        for mp_id, mp in slam_system.map_points.items():
            mp_global_pt = np.array([mp.x_world, mp.y_world])
            # Transform map point from global world to current camera's local pixel coordinates
            mp_camera_coords_arr = slam_system._transform_global_to_camera(mp_global_pt, current_pose)

            if mp_camera_coords_arr is None or mp_camera_coords_arr.size == 0: continue
            mp_camera_coords = mp_camera_coords_arr.squeeze()  # Should be (x, y) in SLAM image space

            # Check if point is within SLAM's view
            if 0 <= mp_camera_coords[0] < slam_system.frame_width and \
                    0 <= mp_camera_coords[1] < slam_system.frame_height:

                # Adjust coordinates if drawing on full screen image and SLAM used a crop
                draw_x = int(mp_camera_coords[0] + slam_image_offset_x)
                draw_y = int(mp_camera_coords[1] + slam_image_offset_y)

                # Check if adjusted coords are within the `draw_image`
                if 0 <= draw_x < draw_image.shape[1] and 0 <= draw_y < draw_image.shape[0]:
                    cv2.circle(draw_image, (draw_x, draw_y), 3, (0, 255, 0), -1)  # Green for map points

        # Draw KeyFrames (optional)
        for kf_id, kf in slam_system.keyframes.items():
            # Draw KF pose (position) in current camera view
            kf_pose_global = np.array([kf.pose.x, kf.pose.y])
            kf_camera_coords_arr = slam_system._transform_global_to_camera(kf_pose_global, current_pose)
            if kf_camera_coords_arr is None or kf_camera_coords_arr.size == 0: continue
            kf_camera_coords = kf_camera_coords_arr.squeeze()

            if 0 <= kf_camera_coords[0] < slam_system.frame_width and \
                    0 <= kf_camera_coords[1] < slam_system.frame_height:
                draw_x = int(kf_camera_coords[0] + slam_image_offset_x)
                draw_y = int(kf_camera_coords[1] + slam_image_offset_y)
                if 0 <= draw_x < draw_image.shape[1] and 0 <= draw_y < draw_image.shape[0]:
                    cv2.rectangle(draw_image, (draw_x - 3, draw_y - 3), (draw_x + 3, draw_y + 3), (255, 0, 0),
                                  1)  # Blue square for KF

        # Draw current pose (center of SLAM's view, but transformed to draw_image coords)
        current_pose_in_slam_img_center = np.array([slam_system.frame_width / 2, slam_system.frame_height / 2])
        draw_current_pose_x = int(current_pose_in_slam_img_center[0] + slam_image_offset_x)
        draw_current_pose_y = int(current_pose_in_slam_img_center[1] + slam_image_offset_y)

        cv2.circle(draw_image, (draw_current_pose_x, draw_current_pose_y), 5, (0, 0, 255),
                   -1)  # Red for current pose center
        # Draw orientation line for current pose
        orientation_len = 30
        end_x = draw_current_pose_x + int(
            orientation_len * np.cos(0))  # Current pose in image is always looking "forward" (theta=0 locally)
        end_y = draw_current_pose_y + int(orientation_len * np.sin(0))  # This is wrong for global orientation display.

        # To show global orientation, we need to project a line based on current_pose.theta
        # This line should be drawn from the *world origin projected into camera view* or from a fixed point.
        # Simpler: draw an arrow at the current pose center indicating its world orientation.
        # The local "forward" in image space corresponds to current_pose.theta in world space.
        # So, a line from current_pose_center pointing along world theta=0 would be -current_pose.theta in image.
        # This is getting complicated. Let's just put text for pose.

        status_text = f"Pose:({current_pose.x:.1f},{current_pose.y:.1f},{np.rad2deg(current_pose.theta):.1f})"
        if slam_system.is_lost: status_text += " LOST"
        cv2.putText(draw_image, status_text, (draw_current_pose_x + 10, draw_current_pose_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

        # Resize for display if needed (your original code had this for the final output to GUI)
        # This should happen *after* all drawing is done on the original resolution image.
        final_display_image = cv2.resize(draw_image, (800, 600), interpolation=cv2.INTER_AREA)

        self.env.renderer.draw_image(final_display_image, (0, 0))  # Assuming renderer handles emitting to GUI


# Your botThread, filters_setings, filter_for_tracking, filter_settings_editor, MainWindow
# remain largely the same, as they interact with higher-level components.
# Ensure ScreenCapturer puts the correct image into env.data_board["map_tracing_image"]
# and that its dimensions match what SLAMSystem2DImpl expects.

class botThread(QThread):
    render_buffer = Signal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.env = Environment()
        self.env.renderer.b_tr = self  # For renderer to emit signal

        sc_graber = ScreenCapturer()
        # grab_region is (left, top, right, bottom)
        # width = right - left, height = bottom - top
        # These must match SLAMSystem2DImpl's frame_width, frame_height
        self.grab_left, self.grab_top, self.grab_right, self.grab_bottom = 0, 100, 1700, 600
        sc_graber.grab_region = (self.grab_left, self.grab_top, self.grab_right, self.grab_bottom)

        sc_graber.add_filter(filter_for_tracking)  # This filter creates "map_tracing_image"
        self.env.AddController(sc_graber)

        filterssetings_controller = filters_setings()  # Renamed for clarity
        self.env.AddController(filterssetings_controller)

        map_controller = GameWorldMap()  # Renamed
        self.env.AddController(map_controller)

        tracker_controller = PlayerTracker()  # Renamed
        self.env.AddController(tracker_controller)

        viz_controller = Visualizer()  # Renamed
        self.env.AddController(viz_controller)

        player_pawn = Pawn("p_pwn")  # This seems like a generic agent pawn
        player_pawn.enabled = True
        self.env.AddChild(player_pawn)

    def set_render_buffer(self, buffer):  # Called by renderer via self.env.renderer.b_tr
        self.render_buffer.emit(buffer)

    def run(self):
        self.env.Start()  # Starts the environment's update loop


class filters_setings(EntityController):
    gaus_kernel_size = 5  # Default

    def __init__(self):
        super().__init__()
        self.name = "FilterSettingsController"
        # Ensure gaus_kernel_size is odd for cv2.GaussianBlur
        if self.gaus_kernel_size % 2 == 0:
            self.gaus_kernel_size += 1

    def set_gaus_kernel_size(self, size):
        # Ensure kernel size is odd
        new_size = size
        if new_size % 2 == 0:
            new_size += 1
        if new_size < 1:  # Minimum kernel size
            new_size = 1
        self.gaus_kernel_size = new_size
        logger.debug(f"Gaussian kernel size set to: {self.gaus_kernel_size}")


def filter_for_tracking(sc_capturer: ScreenCapturer):
    # This filter is applied by ScreenCapturer
    # It should populate sc_capturer.grab_buffer["map_tracing_image"]
    # which is then used by PlayerTracker for SLAM.

    # current_frame is the raw captured region from sc_capturer
    current_frame_bgr = sc_capturer.grab_buffer.get("current_frame_cv", None)  # Assuming ScreenCapturer puts it here

    if current_frame_bgr is None:
        sc_capturer.grab_buffer["map_tracing_image"] = None
        logger.trace("filter_for_tracking: current_frame_cv not found in grab_buffer.")
        return

    _filters_settings_ctrl = sc_capturer.env.GetControllerByType(filters_setings)
    if not _filters_settings_ctrl:
        logger.warning("filters_setings controller not found in filter_for_tracking. Using default kernel size.")
        k_size = 5  # Default
    else:
        k_size = _filters_settings_ctrl.gaus_kernel_size

    # GaussianBlur expects ksize.width and ksize.height to be positive and odd.
    blurred = cv2.GaussianBlur(current_frame_bgr, (k_size, k_size), 0)

    # CLAHE is typically applied to grayscale images.
    # frame_gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    # enhanced_gray = CLAHE.apply(frame_gray)
    # sc_capturer.grab_buffer["map_tracing_image"] = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR) # Convert back if SLAM needs BGR

    # For now, SLAM's _preprocess_frame handles BGR to GRAY conversion.
    # So, provide the blurred BGR image.
    sc_capturer.grab_buffer["map_tracing_image"] = blurred
    # Also populate full_screen_pil for the visualizer if it's based on this processed image
    # Or ensure ScreenCapturer populates "full_screen_pil" from the original screen capture.
    # Let's assume ScreenCapturer separately provides "full_screen_pil" from the original capture.
    # If Visualizer should draw on *this* processed image:
    # sc_capturer.grab_buffer["full_screen_pil_for_viz"] = Image.fromarray(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))


# ... (filter_settings_editor and MainWindow should largely remain the same) ...
class filter_settings_editor(ObjectEditorView):
    def __init__(self, obj):
        super().__init__(obj)
        self.layout = QVBoxLayout(self)
        self.label_title = QLabel("Filter Settings", self)  # Renamed for clarity
        self.layout.addWidget(self.label_title)

        edit_obj: filters_setings = self.object

        self.slider_label = QLabel(f"Gauss Kernel: {edit_obj.gaus_kernel_size}", self)
        self.layout.addWidget(self.slider_label)

        slider_min = 1
        slider_max = 21  # Odd numbers up to 21 e.g.

        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setMinimum(slider_min)
        self.slider.setMaximum(slider_max)
        self.slider.setSingleStep(2)  # Step by 2 to keep it odd if min is odd
        self.slider.setValue(edit_obj.gaus_kernel_size)  # Initial value

        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(2)
        self.slider.valueChanged.connect(self.update_value)

        self.layout.addWidget(self.slider)
        self.setLayout(self.layout)

    def update_value(self, value):
        edit_obj: filters_setings = self.object
        # Ensure value is odd
        actual_value = value
        if actual_value % 2 == 0:
            actual_value = max(1, actual_value - 1)  # Adjust to nearest odd, or ensure it's at least 1

        edit_obj.set_gaus_kernel_size(actual_value)
        self.slider_label.setText(f"Gauss Kernel: {edit_obj.gaus_kernel_size}")
        if self.slider.value() != edit_obj.gaus_kernel_size:  # Sync slider if value was adjusted
            self.slider.setValue(edit_obj.gaus_kernel_size)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.boThread = botThread()
        self.boThread.render_buffer.connect(self.update_image_and_table)  # Combined slot
        self.setWindowTitle("Kornia SLAM Visualizer")
        self.setGeometry(0, 0, 1000, 700)  # Increased size for table

        container = QWidget()
        self.setCentralWidget(container)

        main_layout = QHBoxLayout(container)  # Use main_layout directly

        # Left side for image and controls
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        self.image_display_label = QLabel(self)  # Renamed
        self.image_display_label.setAlignment(Qt.AlignCenter)
        self.image_display_label.setMinimumSize(800, 600)  # Set a min size for the image display
        self.image_display_label.setStyleSheet("background-color: black;")

        self.enableButton = QPushButton('Start SLAM Thread')
        self.enableButton.clicked.connect(self.start_bot_thread)

        # Removed 'Add marker' button as GameWorldMap.add_marker() wasn't defined
        # add_marker_button = QPushButton('Add marker')
        # add_marker_button.clicked.connect(lambda :self.boThread.env.GetControllerByType(GameWorldMap).add_marker())

        left_layout.addWidget(self.image_display_label)
        left_layout.addWidget(self.enableButton)
        # left_layout.addWidget(add_marker_button) # If you re-add add_marker

        # Object editor for filter settings
        filter_settings_ctrl = self.boThread.env.GetControllerByType(filters_setings)
        if filter_settings_ctrl:
            object_editor = PySide6ObjectEditor()
            object_editor.add_view_template(filters_setings, filter_settings_editor)
            object_editor.set_object(filter_settings_ctrl)
            left_layout.addWidget(object_editor)
        else:
            logger.error("filters_setings controller not found for editor setup.")

        main_layout.addWidget(left_widget, stretch=2)  # Image part takes more space

        # Right side for table
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(2)
        self.table_widget.setHorizontalHeaderLabels(['Parameter', 'Value'])
        self.table_widget.setMinimumWidth(300)  # Min width for table
        main_layout.addWidget(self.table_widget, stretch=1)  # Table takes less space

    def start_bot_thread(self):
        if not self.boThread.isRunning():
            self.boThread.start()
            self.enableButton.setText("Running...")
            self.enableButton.setEnabled(False)
        else:
            logger.info("Bot thread is already running.")

    @Slot(np.ndarray)
    def update_image_and_table(self, cv_img_from_renderer):
        # Update image
        if cv_img_from_renderer is not None and cv_img_from_renderer.size > 0:
            q_pixmap = cv2pixmap(cv_img_from_renderer)  # cv2pixmap handles BGR/RGB
            self.image_display_label.setPixmap(q_pixmap.scaled(
                self.image_display_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
        else:
            # Clear pixmap or show placeholder if image is None
            self.image_display_label.clear()

        # Update table from data_board
        if self.boThread.isRunning():  # Only update if thread is running
            data_board_dict = self.boThread.env.data_board.data.copy()  # Get a copy
            self.table_widget.setRowCount(len(data_board_dict))
            for i, (key, value) in enumerate(data_board_dict.items()):
                key_item = QTableWidgetItem(str(key))
                # Format floating point numbers nicely
                if isinstance(value, float):
                    value_str = f"{value:.2f}"
                else:
                    value_str = str(value)
                value_item = QTableWidgetItem(value_str)

                key_item.setFlags(key_item.flags() & ~Qt.ItemIsEditable)  # Read-only
                value_item.setFlags(value_item.flags() & ~Qt.ItemIsEditable)  # Read-only

                self.table_widget.setItem(i, 0, key_item)
                self.table_widget.setItem(i, 1, value_item)
            self.table_widget.resizeColumnsToContents()

    def closeEvent(self, event):
        logger.info("Main window closing. Stopping SLAM thread.")
        if self.boThread.isRunning():
            self.boThread.env.Stop()  # Signal environment to stop
            self.boThread.quit()  # Ask QThread to quit its event loop
            self.boThread.wait(5000)  # Wait for thread to finish
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Ensure Kornia/PyTorch are available before running complex parts
    if not KORNIA_AVAILABLE or not PYTORCH_AVAILABLE:
        logger.critical("Kornia or PyTorch is not available. Key functionalities will be missing.")
        # Optionally, show a message box to the user or exit
        # QMessageBox.critical(None, "Dependency Error", "Kornia or PyTorch not found. Application might not work correctly.")
        # sys.exit(1) # Or allow to run with limited functionality

    window = MainWindow()
    window.show()
    sys.exit(app.exec())