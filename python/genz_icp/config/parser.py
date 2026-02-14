from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# Đảm bảo import đúng GenZConfig từ module C++ đã bind
# from genz_icp.genz_icp_pybind import _GenZConfig as GenZConfig
from genz_icp.genz_icp import GenZConfig
from .config import AdaptiveThresholdConfig, DataConfig, MappingConfig, RegistrationConfig


class GenZPipelineConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="genz_icp_")
    out_dir: str = "results"
    data: DataConfig = DataConfig()
    mapping: MappingConfig = MappingConfig()
    registration: RegistrationConfig = RegistrationConfig()
    adaptive_threshold: AdaptiveThresholdConfig = AdaptiveThresholdConfig()


def _yaml_source(config_file: Optional[Path]) -> Dict[str, Any]:
    if config_file is None:
        return {}
    try:
        yaml = importlib.import_module("yaml")
    except ModuleNotFoundError:
        return {}
        
    with open(config_file) as cfg_file:
        return yaml.safe_load(cfg_file) or {}

def load_config(config_file: Optional[Path]) -> GenZPipelineConfig:
    """Load configuration from an optional yaml file."""
    # Load từ file
    file_config_dict = _yaml_source(config_file)
    
    # [FIX] Pydantic V2 cần cẩn thận khi merge dict lồng nhau
    # Cách đơn giản: Khởi tạo mặc định rồi update nếu cần
    # Hoặc để Pydantic tự lo (như code cũ của bạn cũng tạm ổn)
    config = GenZPipelineConfig(**file_config_dict)

    if config.data.max_range < config.data.min_range:
        config.data.min_range = 0.0

    if config.mapping.voxel_size is None:
        config.mapping.voxel_size = float(config.data.max_range / 100.0)

    return config

def to_genz_config(config: GenZPipelineConfig) -> GenZConfig:
    assert config.mapping.voxel_size is not None, "Voxel size has not been computed!"
    
    c_config = GenZConfig()
    
    # 1. Gán các tham số cũ
    c_config.max_range = config.data.max_range
    c_config.min_range = config.data.min_range
    c_config.map_cleanup_radius = config.mapping.map_cleanup_radius
    c_config.max_points_per_voxel = config.mapping.max_points_per_voxel
    c_config.voxel_size = config.mapping.voxel_size
    c_config.desired_num_voxelized_points = config.mapping.desired_num_voxelized_points
    c_config.min_motion_th = config.adaptive_threshold.min_motion_th
    c_config.initial_threshold = config.adaptive_threshold.initial_threshold
    c_config.planarity_threshold = config.adaptive_threshold.planarity_threshold
    c_config.deskew = config.data.deskew
    c_config.max_num_iterations = config.registration.max_num_iterations
    c_config.convergence_criterion = config.registration.convergence_criterion
    
    # 2. [THÊM MỚI] Gán các tham số Adaptive & Mode
    c_config.use_adaptive_planarity = config.adaptive_threshold.use_adaptive_planarity
    c_config.adaptive_threshold_base = config.adaptive_threshold.adaptive_threshold_base
    c_config.min_adaptive_threshold = config.adaptive_threshold.min_adaptive_threshold
    c_config.max_adaptive_threshold = config.adaptive_threshold.max_adaptive_threshold
    
    # [QUAN TRỌNG] Truyền mode xuống C++
    c_config.registration_mode = config.registration.registration_mode

    return c_config

def write_config(config: GenZPipelineConfig = GenZPipelineConfig(), filename: str = "genz_icp.yaml"):
    with open(filename, "w") as outfile:
        try:
            yaml = importlib.import_module("yaml")
            # Pydantic v2 dùng model_dump()
            yaml.dump(config.model_dump(), outfile, default_flow_style=False)
        except ModuleNotFoundError:
            outfile.write(str(config.model_dump()))
