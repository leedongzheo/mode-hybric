from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np

# [IMPORT CHUẨN] Theo đúng cấu trúc của bạn
from genz_icp.pybind import genz_icp_pybind


@dataclass
class GenZConfig:
    max_range: float = 100.0
    min_range: float = 0.5
    map_cleanup_radius: float = 400.0
    max_points_per_voxel: int = 1
    voxel_size: float = 0.25
    desired_num_voxelized_points: int = 2000
    min_motion_th: float = 0.1
    initial_threshold: float = 2.0
    
    # [CŨ] Dùng cho Baseline
    planarity_threshold: float = 0.2 
    
    # === [THÊM MỚI] Adaptive Threshold ===
    use_adaptive_planarity: bool = True
    adaptive_threshold_base: float = 0.06
    min_adaptive_threshold: float = 0.001
    max_adaptive_threshold: float = 0.2
    # =====================================

    # === [THÊM MỚI] Mode chạy ===
    # 0: Hybrid
    # 1: Point-to-Point
    # 2: Point-to-Plane
    registration_mode: int = 0
    # ============================

    deskew: bool = False
    max_num_iterations: int = 150
    convergence_criterion: float = 0.0001

    def _to_cpp(self):
        # Tạo object config C++
        config = genz_icp_pybind._GenZConfig()
        
        # Gán giá trị thủ công để đảm bảo an toàn và đúng kiểu dữ liệu
        config.max_range = self.max_range
        config.min_range = self.min_range
        config.map_cleanup_radius = self.map_cleanup_radius
        config.max_points_per_voxel = self.max_points_per_voxel
        config.voxel_size = self.voxel_size
        config.desired_num_voxelized_points = self.desired_num_voxelized_points
        config.min_motion_th = self.min_motion_th
        config.initial_threshold = self.initial_threshold
        config.planarity_threshold = self.planarity_threshold
        
        # Adaptive Params
        config.use_adaptive_planarity = self.use_adaptive_planarity
        config.adaptive_threshold_base = self.adaptive_threshold_base
        config.min_adaptive_threshold = self.min_adaptive_threshold
        config.max_adaptive_threshold = self.max_adaptive_threshold
        
        # Mode & Deskew & Solver
        config.registration_mode = self.registration_mode
        config.deskew = self.deskew
        config.max_num_iterations = self.max_num_iterations
        config.convergence_criterion = self.convergence_criterion
        
        return config


def _to_cpp_points(frame: np.ndarray):
    points = np.asarray(frame, dtype=np.float64)
    return genz_icp_pybind._Vector3dVector(points)


class GenZICP:
    def __init__(self, config: Optional[GenZConfig] = None):
        self.config = config or GenZConfig()
        self._odometry = genz_icp_pybind._GenZICP(self.config._to_cpp())

    def register_frame(
        self, frame: np.ndarray, timestamps: Optional[Iterable[float]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        points = _to_cpp_points(frame)
        if timestamps is None:
            # Nhận về tuple (planar, non_planar) từ C++
            planar, non_planar = self._odometry._register_frame(points)
        else:
            planar, non_planar = self._odometry._register_frame(points, list(timestamps))
            
        # Chuyển về numpy array để dễ dùng trong Python
        return np.asarray(planar), np.asarray(non_planar)

    @property
    def poses(self) -> List[np.ndarray]:
        return [np.asarray(pose) for pose in self._odometry._poses()]

    @property
    def last_pose(self) -> np.ndarray:
        return np.asarray(self._odometry._last_pose())

    @property
    def local_map(self) -> np.ndarray:
        return np.asarray(self._odometry._local_map())
    # === [THÊM MỚI] Lấy dữ liệu thời gian từ C++ ===
    @property
    def search_time(self) -> float: 
        return self._odometry._get_search_time()

    @property
    def pca_time(self) -> float: 
        return self._odometry._get_pca_time()

    @property
    def opt_time(self) -> float: 
        return self._odometry._get_opt_time()
    # ===============================================

def voxel_down_sample(frame: np.ndarray, voxel_size: float) -> np.ndarray:
    return np.asarray(genz_icp_pybind._voxel_down_sample(_to_cpp_points(frame), voxel_size))
