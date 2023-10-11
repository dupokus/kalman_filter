#pragma once

#include <Eigen/Dense>

class KalmanFilter 
{
	public:
		KalmanFilter(const Eigen::VectorXf& initial_state, const Eigen::MatrixXf& initial_covariance);

		void predict(const Eigen::MatrixXf& transition_matrix, const Eigen::MatrixXf& process_noise_covariance);

		void update(const Eigen::VectorXf& measurement, const Eigen::MatrixXf& measurement_noise_covariance);

		const Eigen::VectorXf& get_state() const;

		const Eigen::MatrixXf& get_covariance() const;

	private:
		Eigen::VectorXf state_;
		Eigen::MatrixXf covariance_;
};