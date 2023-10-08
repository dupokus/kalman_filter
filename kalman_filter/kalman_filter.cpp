#include <iostream>
#include <Eigen/Dense>
#include <map>

using namespace Eigen;

std::map<int, double> chi2inv95 =
{
    {1, 3.8415},
    {2, 5.9915},
    {3, 7.8147},
    {4, 9.4877},
    {5, 11.070},
    {6, 12.592},
    {7, 14.067},
    {8, 15.507},
    {9, 16.919}
};

class KalmanFilter
{
public:
    MatrixXd motion_mat, update_mat;

    KalmanFilter()
    {
        int ndim = 4;
        double dt = 1.0;

        motion_mat = MatrixXd::Identity(2 * ndim, 2 * ndim);
        for (int i = 0; i < ndim; ++i)
        {
            motion_mat(i, ndim + i) = dt;
        }
        update_mat = MatrixXd::Identity(ndim, 2 * ndim);

        std_weight_position = 1.0 / 20;
        std_weight_velocity = 1.0 / 160;
    }
    std::pair<VectorXd, MatrixXd> initiate(const VectorXd& measurement)
    {
        VectorXd mean_pos = measurement;
        VectorXd mean_vel = VectorXd::Zero(measurement.size());
        VectorXd mean(8);
        mean << mean_pos, mean_vel;

        VectorXd std(8);
        std << 2 * std_weight_position * measurement(3),
            2 * std_weight_position * measurement(3),
            1e-2,
            2 * std_weight_position * measurement(3),
            10 * std_weight_velocity * measurement(3),
            10 * std_weight_velocity * measurement(3),
            1e-5,
            10 * std_weight_velocity * measurement(3);

        MatrixXd covariance = std.asDiagonal();
        return std::make_pair(mean, covariance);
    }

    std::pair<VectorXd, MatrixXd> predict(const VectorXd& mean, const MatrixXd& covariance)
    {
        VectorXd std_pos(4);
        std_pos << std_weight_position * mean(3),
            std_weight_position* mean(3),
            1e-2,
            std_weight_position* mean(3);

        VectorXd std_vel(4);
        std_vel << std_weight_velocity * mean(3),
            std_weight_velocity* mean(3),
            1e-5,
            std_weight_velocity* mean(3);

        MatrixXd motion_cov = (std_pos.array() * std_pos.array()).matrix().asDiagonal();
        motion_cov.bottomRightCorner(4, 4) = (std_vel.array() * std_vel.array()).matrix().asDiagonal();

        VectorXd new_mean = motion_mat * mean;
        MatrixXd new_covariance = motion_mat * covariance * motion_mat.transpose() + motion_cov;

        return std::make_pair(new_mean, new_covariance);
    }

    std::pair<VectorXd, MatrixXd> project(const VectorXd& mean, const MatrixXd& covariance)
    {
        VectorXd std(4);
        std << std_weight_position * mean(3),
            std_weight_position* mean(3),
            1e-1,
            std_weight_position* mean(3);

        MatrixXd innovation_cov = (std.array() * std.array()).matrix().asDiagonal();

        VectorXd projected_mean = update_mat * mean;
        MatrixXd projected_covariance = update_mat * covariance * update_mat.transpose();

        return std::make_pair(projected_mean, projected_covariance + innovation_cov);
    }

    std::pair<Eigen::VectorXd, Eigen::MatrixXd> update(const Eigen::VectorXd& mean, const Eigen::MatrixXd& covariance, const Eigen::VectorXd& measurement)
    {
        auto result = project(mean, covariance);
        Eigen::VectorXd projected_mean = result.first;
        Eigen::MatrixXd projected_cov = result.second;

        Eigen::LDLT<Eigen::MatrixXd> llt(projected_cov.transpose()); // Transpose here
        Eigen::VectorXd innovation = measurement - projected_mean;

        Eigen::VectorXd kalman_gain = llt.solve(innovation);

        Eigen::VectorXd new_mean = mean + kalman_gain;
        Eigen::MatrixXd new_covariance = covariance - llt.matrixL() * llt.transpose().solve(covariance);

        return std::make_pair(new_mean, new_covariance);
    }

private:
    double std_weight_position;
    double std_weight_velocity;
};

int main()
{
    KalmanFilter kf;
    VectorXd measurement(4);
    measurement << 1, 2, 1.5, 3;

    Eigen::VectorXd mean, covariance;
    std::tie(mean, covariance) = kf.initiate(measurement);
    std::cout << "Initial Mean:\n" << mean << std::endl;
    std::cout << "Initial Covariance:\n" << covariance << std::endl;

    VectorXd new_measurement(4);
    new_measurement << 1.2, 2.2, 1.7, 3.2;

    Eigen::VectorXd updated_mean, updated_covariance;
    std::tie(updated_mean, updated_covariance) = kf.update(mean, covariance, new_measurement);
    std::cout << "Updated Mean:\n" << updated_mean << std::endl;
    std::cout << "Updated Covariance:\n" << updated_covariance << std::endl;

    return 0;
}