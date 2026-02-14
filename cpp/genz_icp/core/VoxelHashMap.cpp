#include "VoxelHashMap.hpp"

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <Eigen/Core>
#include <algorithm>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>
#include <array>
#include <cmath>

namespace {
static const std::array<Eigen::Vector3i, 27> voxel_shifts{
    Eigen::Vector3i(0, 0, 0),   Eigen::Vector3i(1, 0, 0),   Eigen::Vector3i(-1, 0, 0),
    Eigen::Vector3i(0, 1, 0),   Eigen::Vector3i(0, -1, 0),  Eigen::Vector3i(0, 0, 1),
    Eigen::Vector3i(0, 0, -1),  Eigen::Vector3i(1, 1, 0),   Eigen::Vector3i(1, -1, 0),
    Eigen::Vector3i(-1, 1, 0),  Eigen::Vector3i(-1, -1, 0), Eigen::Vector3i(1, 0, 1),
    Eigen::Vector3i(1, 0, -1),  Eigen::Vector3i(-1, 0, 1),  Eigen::Vector3i(-1, 0, -1),
    Eigen::Vector3i(0, 1, 1),   Eigen::Vector3i(0, 1, -1),  Eigen::Vector3i(0, -1, 1),
    Eigen::Vector3i(0, -1, -1), Eigen::Vector3i(1, 1, 1),   Eigen::Vector3i(1, 1, -1),
    Eigen::Vector3i(1, -1, 1),  Eigen::Vector3i(1, -1, -1), Eigen::Vector3i(-1, 1, 1),
    Eigen::Vector3i(-1, 1, -1), Eigen::Vector3i(-1, -1, 1), Eigen::Vector3i(-1, -1, -1)
};
}  // namespace

namespace genz_icp {

// === [HÀM MỚI] Dùng cho Hybrid ICP ===
std::tuple<Eigen::Vector3d, std::vector<Eigen::Vector3d>, double>
VoxelHashMap::GetClosestNeighborAndNeighbors(const Eigen::Vector3d &query) const {
    Eigen::Vector3d closest_neighbor = Eigen::Vector3d::Zero();
    double closest_distance = std::numeric_limits<double>::max();
    std::vector<Eigen::Vector3d> neighbors;

    // Tính chỉ số Voxel của điểm query
    auto kx = static_cast<int>(std::floor(query.x() / voxel_size_));
    auto ky = static_cast<int>(std::floor(query.y() / voxel_size_));
    auto kz = static_cast<int>(std::floor(query.z() / voxel_size_));

    // Duyệt 27 hàng xóm
    for (const auto &shift : voxel_shifts) {
        Voxel voxel(kx + shift.x(), ky + shift.y(), kz + shift.z());
        auto search = map_.find(voxel);
        
        if (search != map_.end()) {
            const auto &points = search->second.points;
            
            // Lấy hết điểm làm hàng xóm để tính PCA
            neighbors.insert(neighbors.end(), points.begin(), points.end());

            // Tìm điểm gần nhất (Nearest Neighbor) để làm Target cho ICP
            for (const auto &pt : points) {
                double dist = (pt - query).squaredNorm(); // Dùng squaredNorm cho nhanh
                if (dist < closest_distance) {
                    closest_distance = dist;
                    closest_neighbor = pt;
                }
            }
        }
    }
    return {closest_neighbor, neighbors, std::sqrt(closest_distance)};
}
// =====================================

std::vector<Eigen::Vector3d> VoxelHashMap::Pointcloud() const {
    std::vector<Eigen::Vector3d> points;
    points.reserve(max_points_per_voxel_ * map_.size());
    for (const auto &[voxel, voxel_block] : map_) {
        (void)voxel;
        for (const auto &point : voxel_block.points) {
            points.emplace_back(point);
        }
    }
    return points;
}

void VoxelHashMap::Update(const std::vector<Eigen::Vector3d> &points, const Eigen::Vector3d &origin) {
    AddPoints(points);
    RemovePointsFarFromLocation(origin);
}

void VoxelHashMap::Update(const std::vector<Eigen::Vector3d> &points, const Sophus::SE3d &pose) {
    std::vector<Eigen::Vector3d> points_transformed(points.size());
    std::transform(points.cbegin(), points.cend(), points_transformed.begin(),
                   [&](const auto &point) { return pose * point; });
    const Eigen::Vector3d &origin = pose.translation();
    Update(points_transformed, origin);
}

void VoxelHashMap::AddPoints(const std::vector<Eigen::Vector3d> &points) {
    std::for_each(points.cbegin(), points.cend(), [&](const auto &point) {
        auto k = (point / voxel_size_).template cast<int>();
        auto voxel = Voxel(k.x(), k.y(), k.z());
        
        auto search = map_.find(voxel);
        if (search != map_.end()) {
            auto &voxel_block = search.value();
            voxel_block.AddPoint(point);
        } else {
            map_.emplace(voxel, VoxelBlock(point, max_points_per_voxel_));
        }
    });
}

void VoxelHashMap::RemovePointsFarFromLocation(const Eigen::Vector3d &origin) {
    const auto max_distance2 = map_cleanup_radius_ * map_cleanup_radius_;
    for (auto it = map_.begin(); it != map_.end();) {
        if (it->second.points.empty()) {
            it = map_.erase(it);
            continue;
        }
        if ((it->second.points.front() - origin).squaredNorm() > max_distance2) {
            it = map_.erase(it);
        } else {
            ++it;
        }
    }
}

} // namespace genz_icp