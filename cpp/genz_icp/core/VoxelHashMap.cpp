// MIT License
// ... (Giữ nguyên phần License của bạn)

#include "VoxelHashMap.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>
#include <array>
#include <cmath>

namespace {
// Voxel shifts for 27-neighbor search
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

// === [HÀM CỐT LÕI CHO HYBRID & ADAPTIVE] Đã fix lỗi tọa độ ===
std::tuple<Eigen::Vector3d, std::vector<Eigen::Vector3d>, double>
VoxelHashMap::GetClosestNeighborAndNeighbors(const Eigen::Vector3d &query) const {
    Eigen::Vector3d closest_neighbor = Eigen::Vector3d::Zero();
    double closest_squared_distance = std::numeric_limits<double>::max();
    std::vector<Eigen::Vector3d> neighbors;

    // SỬA LỖI CHÍ MẠNG: Dùng ép kiểu int (Truncation) giống hệt bản GỐC và hàm AddPoints
    auto kx = static_cast<int>(query.x() / voxel_size_);
    auto ky = static_cast<int>(query.y() / voxel_size_);
    auto kz = static_cast<int>(query.z() / voxel_size_);

    // Duyệt 27 voxel lân cận
    for (const auto &shift : voxel_shifts) {
        Voxel voxel(kx + shift.x(), ky + shift.y(), kz + shift.z());
        auto search = map_.find(voxel);
        
        if (search != map_.end()) {
            const auto &points = search->second.points;
            
            // Lấy toàn bộ điểm làm lân cận cho PCA
            neighbors.insert(neighbors.end(), points.begin(), points.end());

            // Tìm điểm gần nhất (Nearest Neighbor) bằng squaredNorm để tối ưu tốc độ
            for (const auto &pt : points) {
                double dist_sq = (pt - query).squaredNorm(); 
                if (dist_sq < closest_squared_distance) {
                    closest_squared_distance = dist_sq;
                    closest_neighbor = pt;
                }
            }
        }
    }
    
    // Trả về khoảng cách thực để so sánh max_correspondence_distance bên file Registration.cpp
    double closest_distance = (closest_squared_distance < std::numeric_limits<double>::max()) 
                            ? std::sqrt(closest_squared_distance) 
                            : std::numeric_limits<double>::max();

    return {closest_neighbor, neighbors, closest_distance};
}
// =============================================================

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
        // Đồng bộ Truncation với khâu Search
        auto k = (point / voxel_size_).template cast<int>();
        auto voxel = Voxel(k.x(), k.y(), k.z());
        auto search = map_.find(voxel);
        
        if (search != map_.end()) {
            auto &voxel_block = search.value();
            voxel_block.AddPoint(point);
        } else {
            map_.insert({voxel, VoxelBlock(point, max_points_per_voxel_)});
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
        // Check distance of the first point in the block
        if ((it->second.points.front() - origin).squaredNorm() > (max_distance2)) {
            it = map_.erase(it);
        } else {
            ++it;
        }
    }
}

} // namespace genz_icp
