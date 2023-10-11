#include <iostream>
#include <Eigen/Dense>

const Eigen::MatrixXf I = Eigen::MatrixXf::Identity(2, 2);

class KalmanFilter {
public:
    KalmanFilter(const Eigen::VectorXf& initial_state, const Eigen::MatrixXf& initial_covariance)
        : state_(initial_state), covariance_(initial_covariance) {
    }

    void predict(const Eigen::MatrixXf& transition_matrix, const Eigen::MatrixXf& process_noise_covariance) {
        state_ = transition_matrix * state_;
        covariance_ = transition_matrix * covariance_ * transition_matrix.transpose() + process_noise_covariance;
    }

    void update(const Eigen::VectorXf& measurement, const Eigen::MatrixXf& measurement_noise_covariance) {
        // Calculate the Kalman gain.
        Eigen::MatrixXf kalman_gain = covariance_ * measurement_noise_covariance.transpose() * (measurement_noise_covariance + covariance_).inverse();

        // Update the state.
        state_ += kalman_gain * (measurement - state_);

        // Update the covariance.
        covariance_ = (I - kalman_gain * measurement_noise_covariance) * covariance_;
    }

    const Eigen::VectorXf& get_state() const {
        return state_;
    }

    const Eigen::MatrixXf& get_covariance() const {
        return covariance_;
    }

private:
    Eigen::VectorXf state_;
    Eigen::MatrixXf covariance_;
};

int main() 
{
    // Create a Kalman filter with initial state and covariance.
    Eigen::VectorXf initial_state(2);
    initial_state << 0, 0;
    Eigen::MatrixXf initial_covariance(2, 2);
    initial_covariance << 1, 0,
        0, 1;
    KalmanFilter kalman_filter(initial_state, initial_covariance);

    // Predict the next state.
    Eigen::MatrixXf transition_matrix(2, 2);
    transition_matrix << 1, 1,
        0, 1;
    Eigen::MatrixXf process_noise_covariance(2, 2);
    process_noise_covariance << 0.1, 0,
        0, 0.1;
    kalman_filter.predict(transition_matrix, process_noise_covariance);

    // Update the state with a measurement.
    Eigen::VectorXf measurement(2);
    measurement << 1, 1;
    Eigen::MatrixXf measurement_noise_covariance(2, 2);
    measurement_noise_covariance << 0.01, 0,
        0, 0.01;
    kalman_filter.update(measurement, measurement_noise_covariance);

    // Get the updated state and covariance.
    Eigen::VectorXf updated_state = kalman_filter.get_state();
    Eigen::MatrixXf updated_covariance = kalman_filter.get_covariance();

    // Print the updated state.
    std::cout << "Updated state: " << updated_state << std::endl;

    return 0;
}
