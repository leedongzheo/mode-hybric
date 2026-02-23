from pydantic import BaseModel
from typing import Optional

class DataConfig(BaseModel):
    max_range: float = 100.0
    # min_range: float = 0.5
    min_range: float = 0.0
    deskew: bool = True

class MappingConfig(BaseModel):
    voxel_size: Optional[float] = None
    # map_cleanup_radius: float = 100.0
    map_cleanup_radius: float = 400
    # max_points_per_voxel: int = 1
    max_points_per_voxel: int = 20
    desired_num_voxelized_points: int = 2000

class RegistrationConfig(BaseModel):
    # max_num_iterations: int = 100
    # max_num_iterations: int = 150
    max_num_iterations: int = 500
    convergence_criterion: float = 0.0001
    
    # [THÊM MỚI] Chế độ chạy:
    # 0: Hybrid (Adaptive) - Default
    # 1: Point-to-Point Only
    # 2: Point-to-Plane Only
    registration_mode: int = 1

class AdaptiveThresholdConfig(BaseModel):
    initial_threshold: float = 2.0
    min_motion_th: float = 0.1
    
    # [CŨ] Dùng cho Baseline
    planarity_threshold: float = 0.07
    
    # [THÊM MỚI] Dùng cho Adaptive
    use_adaptive_planarity: bool = True
    adaptive_threshold_base: float = 0.07
    min_adaptive_threshold: float = 0.01
    max_adaptive_threshold: float = 0.2
