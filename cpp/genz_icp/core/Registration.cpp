#include "Registration.hpp"

#include <Eigen/Eigenvalues>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/enumerable_thread_specific.h>

#include <algorithm>
#include <cmath>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <tuple>
#include <iostream>
#include <iomanip> 
#include <chrono> // [THÊM MỚI] Dùng để đo thời gian

namespace Eigen {
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix3_6d = Eigen::Matrix<double, 3, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
}  // namespace Eigen

namespace {

inline double square(double x) { return x * x; }

// --- Hybrid Correspondence Struct ---
struct HybridCorrespondence {
    std::vector<Eigen::Vector3d> src_planar, tgt_planar, normals;
    std::vector<Eigen::Vector3d> src_non_planar, tgt_non_planar;
    size_t planar_count = 0, non_planar_count = 0;
    
    // [THÊM MỚI] Biến lưu thời gian CPU nội bộ
    double cpu_time_search = 0.0;
    double cpu_time_pca = 0.0;
};

// --- Adaptive Threshold Functions ---
double ComputeAdaptivePlanarityThreshold(const std::vector<Eigen::Vector3d>& neighbors, double base, double min_thr, double max_thr){
    constexpr double ref_neighbors = 20.0;
    // Avoid division by zero
    double n = std::max(ref_neighbors, static_cast<double>(neighbors.size()));
    double thr = base * ref_neighbors / n;
    return std::clamp(thr, min_thr, max_thr);
}

std::tuple<bool, Eigen::Vector3d> EstimateNormalAndPlanarity(
    const std::vector<Eigen::Vector3d>& neighbors, 
    double threshold_param, 
    bool use_adaptive,
    double min_thr,
    double max_thr
    )
{
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const auto& pt : neighbors) mean += pt;
    mean /= static_cast<double>(neighbors.size());

    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (const auto& pt : neighbors) {
        Eigen::Vector3d d = pt - mean;
        cov.noalias() += d * d.transpose();
    }
    cov /= static_cast<double>(neighbors.size());

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(cov);
    const auto& evals = eig.eigenvalues();
    const auto& evecs = eig.eigenvectors();

    const double lambda0 = evals(0);
    const double sumlam  = evals(0) + evals(1) + evals(2) + 1e-12;
    const double planarity = lambda0 / sumlam;

    double final_threshold;
    if (use_adaptive) {
        final_threshold = ComputeAdaptivePlanarityThreshold(neighbors, threshold_param, min_thr, max_thr);
    } else {
        final_threshold = threshold_param;
    }
    
    const bool is_planar = planarity < final_threshold;
    Eigen::Vector3d normal = evecs.col(0);
    return {is_planar, normal};
}

// --- Parallel Hybrid Correspondence Search ---
HybridCorrespondence ComputeHybridCorrespondencesParallel(
    const std::vector<Eigen::Vector3d>& source_points,
    const genz_icp::VoxelHashMap& voxel_map,
    double max_correspondence_distance,
    double threshold_param,
    bool use_adaptive,
    double min_thr,
    double max_thr,
    int registration_mode // 0: Hybrid, 1: Point-to-Point, 2: Point-to-Plane
    )
{
    struct LocalBuf {
        std::vector<Eigen::Vector3d> src_planar, tgt_planar, normals;
        std::vector<Eigen::Vector3d> src_non_planar, tgt_non_planar;
        size_t planar_count = 0, non_planar_count = 0;
        
        // [THÊM MỚI] Biến lưu thời gian thread cục bộ
        double cpu_time_search = 0.0;
        double cpu_time_pca = 0.0;

        void reserve_hint(size_t n) {
            const size_t hint = std::max<size_t>(32, n / 2);
            src_planar.reserve(hint);  tgt_planar.reserve(hint); normals.reserve(hint);
            src_non_planar.reserve(hint); tgt_non_planar.reserve(hint);
        }
    };

    tbb::enumerable_thread_specific<LocalBuf> tls;
    for (auto it = tls.begin(); it != tls.end(); ++it) {
        it->reserve_hint(source_points.size());
    }

    // tbb::parallel_for(
    //     tbb::blocked_range<size_t>(0, source_points.size()),
    //     [&](const tbb::blocked_range<size_t>& r) {
    //         auto& buf = tls.local();
    //         for (size_t i = r.begin(); i != r.end(); ++i) {
    //             const auto& pt = source_points[i];
                
    //             // --- ĐO THỜI GIAN NEIGHBOR SEARCH ---
    //             auto t1_search = std::chrono::high_resolution_clock::now();
    //             auto [closest, neighbors, dist] = voxel_map.GetClosestNeighborAndNeighbors(pt);
    //             auto t2_search = std::chrono::high_resolution_clock::now();
    //             buf.cpu_time_search += std::chrono::duration<double, std::milli>(t2_search - t1_search).count();
    //             // ------------------------------------

    //             if (dist > max_correspondence_distance) continue;

    //             // --- MODE 1: POINT-TO-POINT ONLY ---
    //             // Ignores Adaptive Flag, PCA, Normals
    //             if (registration_mode == 1) {
    //                 buf.src_non_planar.push_back(pt);
    //                 buf.tgt_non_planar.push_back(closest);
    //                 buf.non_planar_count++;
    //                 continue; 
    //             }

    //             // --- MODE 0 (Hybrid) & MODE 2 (Plane-Only) ---
    //             // Needs PCA to check planarity
    //             if (neighbors.size() >= 5) { 
                    
    //                 // --- ĐO THỜI GIAN PCA & PLANARITY ---
    //                 auto t1_pca = std::chrono::high_resolution_clock::now();
    //                 auto [is_planar, normal] = EstimateNormalAndPlanarity(neighbors, threshold_param, use_adaptive, min_thr, max_thr);
    //                 auto t2_pca = std::chrono::high_resolution_clock::now();
    //                 buf.cpu_time_pca += std::chrono::duration<double, std::milli>(t2_pca - t1_pca).count();
    //                 // ------------------------------------
                    
    //                 if (is_planar) {
    //                     // Both Hybrid and Plane-Only accept Planar points
    //                     buf.src_planar.push_back(pt);
    //                     buf.tgt_planar.push_back(closest);
    //                     buf.normals.push_back(normal);
    //                     buf.planar_count++;
    //                 } else {
    //                     // Point is Non-Planar (Tree, Grass, Noise...)
    //                     if (registration_mode == 0) {
    //                         // Hybrid: Keep it, use Point-to-Point constraint
    //                         buf.src_non_planar.push_back(pt);
    //                         buf.tgt_non_planar.push_back(closest);
    //                         buf.non_planar_count++;
    //                     } 
    //                     // Mode 2 (Plane-Only): Discard it (Do nothing)
    //                 }
    //             } else {
    //                 // Not enough neighbors for PCA
    //                 if (registration_mode == 0) {
    //                     // Hybrid: Fallback to Point-to-Point
    //                     buf.src_non_planar.push_back(pt);
    //                     buf.tgt_non_planar.push_back(closest);
    //                     buf.non_planar_count++;
    //                 }
    //                 // Mode 2: Discard
    //             }
    //         }
    //     }
    // );
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, source_points.size()),
        [&](const tbb::blocked_range<size_t>& r) {
            auto& buf = tls.local();
            for (size_t i = r.begin(); i != r.end(); ++i) {
                const auto& pt = source_points[i];
                
                // --- ĐO THỜI GIAN NEIGHBOR SEARCH ---
                auto t1_search = std::chrono::high_resolution_clock::now();
                auto [closest, neighbors, dist] = voxel_map.GetClosestNeighborAndNeighbors(pt);
                auto t2_search = std::chrono::high_resolution_clock::now();
                buf.cpu_time_search += std::chrono::duration<double, std::milli>(t2_search - t1_search).count();
                // ------------------------------------

                if (dist > max_correspondence_distance) continue;

                // --- MODE 1: POINT-TO-POINT ONLY ---
                if (registration_mode == 1) {
                    buf.src_non_planar.push_back(pt);
                    buf.tgt_non_planar.push_back(closest);
                    buf.non_planar_count++;
                    continue; 
                }

                if (neighbors.size() >= 5) { 
                    
                    // --- ĐO THỜI GIAN PCA & PLANARITY ---
                    auto t1_pca = std::chrono::high_resolution_clock::now();
                    auto [is_planar, normal] = EstimateNormalAndPlanarity(neighbors, threshold_param, use_adaptive, min_thr, max_thr);
                    auto t2_pca = std::chrono::high_resolution_clock::now();
                    buf.cpu_time_pca += std::chrono::duration<double, std::milli>(t2_pca - t1_pca).count();
                    // ------------------------------------
                    
                    // === SỬA LOGIC Ở ĐÂY THEO Ý BẠN ===
                    if (registration_mode == 2) {
                        // NẾU LÀ MODE 2: Bất chấp có phẳng hay không, cứ lấy Normal vector
                        // và ép nó trở thành Point-to-Plane (100% số điểm đủ điều kiện).
                        buf.src_planar.push_back(pt);
                        buf.tgt_planar.push_back(closest);
                        buf.normals.push_back(normal);
                        buf.planar_count++;
                    } 
                    else { // MODE 0: HYBRID (Phân loại thông minh)
                        if (is_planar) {
                            buf.src_planar.push_back(pt);
                            buf.tgt_planar.push_back(closest);
                            buf.normals.push_back(normal);
                            buf.planar_count++;
                        } else {
                            buf.src_non_planar.push_back(pt);
                            buf.tgt_non_planar.push_back(closest);
                            buf.non_planar_count++;
                        }
                    }
                } else {
                    // Nếu không đủ 5 điểm để tính PCA -> Bắt buộc phải là Pt2Pt
                    // Vì không có Normal Vector thì không thể dùng công thức Pt2Pl được.
                    if (registration_mode == 0 || registration_mode == 2) {
                        // Thậm chí ở Mode 2, ta đành phải dùng Pt2Pt cho những điểm này 
                        // (Hoặc bạn có thể vứt bỏ chúng đi nếu muốn "thuần khiết" Pt2Pl 100%)
                        // Ở đây ta cứ giữ lại để so sánh công bằng về số lượng điểm.
                        buf.src_non_planar.push_back(pt);
                        buf.tgt_non_planar.push_back(closest);
                        buf.non_planar_count++;
                    }
                }
            }
        }
    );
    // Merge results
    HybridCorrespondence out;
    size_t total_planar = 0, total_nonplanar = 0;
    for (auto& buf : tls) {
        total_planar    += buf.planar_count;
        total_nonplanar += buf.non_planar_count;
        
        // [THÊM MỚI] Gom thời gian CPU lại
        out.cpu_time_search += buf.cpu_time_search;
        out.cpu_time_pca += buf.cpu_time_pca;
    }
    
    out.src_planar.reserve(total_planar);
    out.tgt_planar.reserve(total_planar);
    out.normals.reserve(total_planar);
    out.src_non_planar.reserve(total_nonplanar);
    out.tgt_non_planar.reserve(total_nonplanar);

    for (auto& buf : tls) {
        out.planar_count     += buf.planar_count;
        out.non_planar_count += buf.non_planar_count;
        out.src_planar.insert(out.src_planar.end(), buf.src_planar.begin(), buf.src_planar.end());
        out.tgt_planar.insert(out.tgt_planar.end(), buf.tgt_planar.begin(), buf.tgt_planar.end());
        out.normals.insert(out.normals.end(), buf.normals.begin(), buf.normals.end());
        out.src_non_planar.insert(out.src_non_planar.end(), buf.src_non_planar.begin(), buf.src_non_planar.end());
        out.tgt_non_planar.insert(out.tgt_non_planar.end(), buf.tgt_non_planar.begin(), buf.tgt_non_planar.end());
    }
    return out;
}

void TransformPoints(const Sophus::SE3d &T, std::vector<Eigen::Vector3d> &points) {
    std::transform(points.cbegin(), points.cend(), points.begin(),
                   [&](const auto &point) { return T * point; });
}

// --- GenZ-ICP BuildLinearSystem ---
std::tuple<Eigen::Matrix6d, Eigen::Vector6d> BuildLinearSystem(
    const std::vector<Eigen::Vector3d> &src_planar,
    const std::vector<Eigen::Vector3d> &tgt_planar,
    const std::vector<Eigen::Vector3d> &normals,
    const std::vector<Eigen::Vector3d> &src_non_planar,
    const std::vector<Eigen::Vector3d> &tgt_non_planar,
    double kernel,
    double alpha) {

    struct ResultTuple {
        Eigen::Matrix6d JTJ;
        Eigen::Vector6d JTr;

        ResultTuple() : JTJ(Eigen::Matrix6d::Zero()), JTr(Eigen::Vector6d::Zero()) {}

        ResultTuple operator+(const ResultTuple &other) const {
            ResultTuple result;
            result.JTJ = JTJ + other.JTJ;
            result.JTr = JTr + other.JTr;
            return result;
        }
    };

    auto compute_jacobian_and_residual_planar = [&](auto i) {
        double r_planar = (src_planar[i] - tgt_planar[i]).dot(normals[i]); 
        Eigen::Matrix<double, 1, 6> J_planar; 
        J_planar.block<1, 3>(0, 0) = normals[i].transpose(); 
        J_planar.block<1, 3>(0, 3) = (src_planar[i].cross(normals[i])).transpose();
        return std::make_tuple(J_planar, r_planar);
    };

    auto compute_jacobian_and_residual_non_planar = [&](auto i) {
        const Eigen::Vector3d r_non_planar = src_non_planar[i] - tgt_non_planar[i];
        Eigen::Matrix3_6d J_non_planar;
        J_non_planar.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        J_non_planar.block<3, 3>(0, 3) = -1.0 * Sophus::SO3d::hat(src_non_planar[i]);
        return std::make_tuple(J_non_planar, r_non_planar);
    };

    double kernel_squared = kernel * kernel;
    auto compute = [&](const tbb::blocked_range<size_t> &r, ResultTuple J) -> ResultTuple {
        auto Weight = [&](double residual_squared) {
            return kernel_squared / square(kernel + residual_squared);
        };
        auto &[JTJ_private, JTr_private] = J;
        for (size_t i = r.begin(); i < r.end(); ++i) {
            if (i < src_planar.size()) { 
                const auto &[J_planar, r_planar] = compute_jacobian_and_residual_planar(i);
                double w_planar = Weight(r_planar * r_planar);
                JTJ_private.noalias() += alpha * J_planar.transpose() * w_planar * J_planar;
                JTr_private.noalias() += alpha * J_planar.transpose() * w_planar * r_planar;
            } else { 
                size_t index = i - src_planar.size();
                if (index < src_non_planar.size()) {
                    const auto &[J_non_planar, r_non_planar] = compute_jacobian_and_residual_non_planar(index);
                    const double w_non_planar = Weight(r_non_planar.squaredNorm());
                    JTJ_private.noalias() += (1 - alpha) * J_non_planar.transpose() * w_non_planar * J_non_planar;
                    JTr_private.noalias() += (1 - alpha) * J_non_planar.transpose() * w_non_planar * r_non_planar;
                }
            }
        }
        return J;
    };

    size_t total_size = src_planar.size() + src_non_planar.size();
    const auto &[JTJ, JTr] = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, total_size),
        ResultTuple(),
        compute,
        [](const ResultTuple &a, const ResultTuple &b) {
            return a + b;
        });

    return std::make_tuple(JTJ, JTr);
}

}  // namespace genz_icp

namespace genz_icp {

Registration::Registration(int max_num_iteration, double convergence_criterion)
    : max_num_iterations_(max_num_iteration), 
      convergence_criterion_(convergence_criterion) {}

std::tuple<Sophus::SE3d, std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>> Registration::RegisterFrame(
                                                                                                    const std::vector<Eigen::Vector3d> &frame,
                                                                                                    const VoxelHashMap &voxel_map,
                                                                                                    const Sophus::SE3d &initial_guess,
                                                                                                    double max_correspondence_distance,
                                                                                                    double kernel,
                                                                                                    double adaptive_base,
                                                                                                    bool use_adaptive,
                                                                                                    double min_thr,
                                                                                                    double max_thr,
                                                                                                    int registration_mode // 0:Hybrid, 1:Pt2Pt, 2:Pt2Pl
                                                                                                    ) {
    
    std::vector<Eigen::Vector3d> final_planar_points;
    std::vector<Eigen::Vector3d> final_non_planar_points;
    final_planar_points.clear();
    final_non_planar_points.clear();

    if (voxel_map.Empty()) return std::make_tuple(initial_guess, final_planar_points, final_non_planar_points);

    std::vector<Eigen::Vector3d> source = frame;
    TransformPoints(initial_guess, source);

    Sophus::SE3d T_icp = Sophus::SE3d();
    
    // [THÊM MỚI] Biến tích lũy thời gian cho toàn bộ Frame
    double frame_time_search = 0.0;
    double frame_time_pca = 0.0;
    double frame_time_opt = 0.0;

    for (int j = 0; j < max_num_iterations_; ++j) {
        
        // --- 1. Đo Wall Time cho khâu Search & PCA ---
        auto t_start_corr = std::chrono::high_resolution_clock::now();
        
        auto corr = ComputeHybridCorrespondencesParallel(
            source, 
            voxel_map, 
            max_correspondence_distance, 
            adaptive_base, 
            use_adaptive,
            min_thr, 
            max_thr,
            registration_mode
        );
        
        auto t_end_corr = std::chrono::high_resolution_clock::now();
        double corr_wall_ms = std::chrono::duration<double, std::milli>(t_end_corr - t_start_corr).count();
        
        // --- Phân bổ lại Wall Time dựa trên tỷ lệ CPU Time ---
        double total_cpu = corr.cpu_time_search + corr.cpu_time_pca;
        if (total_cpu > 0.0) {
            frame_time_search += corr_wall_ms * (corr.cpu_time_search / total_cpu);
            frame_time_pca += corr_wall_ms * (corr.cpu_time_pca / total_cpu);
        } else {
            // Nếu không có phép tính nào xảy ra (rất hiếm), đẩy hết vào search
            frame_time_search += corr_wall_ms;
        }
        // ----------------------------------------------

        double total_points = static_cast<double>(corr.planar_count + corr.non_planar_count);
        double alpha = (total_points > 0.0) ? static_cast<double>(corr.planar_count) / total_points : 0.5;
        std::cout << "ICP Iter " << j << " | Alpha = " << std::fixed << std::setprecision(4) << alpha << "\n";
        // --- 2. Đo Wall Time cho ICP Optimizer ---
        auto t_start_opt = std::chrono::high_resolution_clock::now();
        
        // Feed data to the solver
        const auto &[JTJ, JTr] = BuildLinearSystem(
            corr.src_planar, corr.tgt_planar, corr.normals, 
            corr.src_non_planar, corr.tgt_non_planar, 
            kernel, alpha
        );

        const Eigen::Vector6d dx = JTJ.ldlt().solve(-JTr);
        const Sophus::SE3d estimation = Sophus::SE3d::exp(dx);
        TransformPoints(estimation, source);
        T_icp = estimation * T_icp;
        
        auto t_end_opt = std::chrono::high_resolution_clock::now();
        frame_time_opt += std::chrono::duration<double, std::milli>(t_end_opt - t_start_opt).count();
        // -----------------------------------------

        if (dx.norm() < convergence_criterion_ || j == max_num_iterations_ - 1) {
            final_planar_points = corr.src_planar;
            final_non_planar_points = corr.src_non_planar;
            break;
        }
    }
    // Thay vì in ra std::cout, hãy lưu nó lại:
    time_search_ = frame_time_search;
    time_pca_ = frame_time_pca;
    time_opt_ = frame_time_opt;
    // [THÊM MỚI] In kết quả thời gian phân bổ của Frame này ra Terminal
    // std::cout << std::fixed << std::setprecision(3);
    // std::cout << "RUNTIME_LOG|" 
    //           << frame_time_search << "|" 
    //           << frame_time_pca << "|" 
    //           << frame_time_opt << "\n";

    return std::make_tuple(T_icp * initial_guess, final_planar_points, final_non_planar_points);
}

}  // namespace genz_icp
