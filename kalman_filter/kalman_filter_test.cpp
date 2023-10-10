#include <gtest/gtest.h>
#include <Eigen/Dense>

#include "kalman_filter.h"

TEST(KalmanFilterTest, Predict) {
    // Create a Kalman filter with initial state and covariance.
    VectorXf initial_state(2);
    initial_state << 0, 0;
    MatrixXf initial_covariance(2, 2);
    initial_covariance << 1, 0,
        0, 1;
    KalmanFilter kalman_filter(initial_state, initial_covariance);

    // Predict the next state.
    MatrixXf transition_matrix(2, 2);
    transition_matrix << 1, 1,
        0, 1;
    MatrixXf process_noise_covariance(2, 2);
    process_noise_covariance << 0.1, 0,
        0, 0.1;
    kalman_filter.predict(transition_matrix, process_noise_covariance);

    // Expected predicted state.
    VectorXf expected_predicted_state(2);
    expected_predicted_state << 1, 0;

    // Assert that the predicted state is close to the expected value.
    EXPECT_NEAR(kalman_filter.get_state(), expected_predicted_state, 1e-6);
}

TEST(KalmanFilterTest, Update) {
    // Create a Kalman filter with initial state and covariance.
    VectorXf initial_state(2);
    initial_state << 0, 0;
    MatrixXf initial_covariance(2, 2);
    initial_covariance << 1, 0,
        0, 1;
    KalmanFilter kalman_filter(initial_state, initial_covariance);

    // Update the state with a measurement.
    VectorXf measurement(2);
    measurement << 1, 1;
    MatrixXf measurement_noise_covariance(2, 2);
    measurement_noise_covariance << 0.01, 0,
        0, 0.01;
    kalman_filter.update(measurement, measurement_noise_covariance);

    // Expected updated state.
    VectorXf expected_updated_state(2);
    expected_updated_state << 0.99, 0;

    // Assert that the updated state is close to the expected value.
    EXPECT_NEAR(kalman_filter.get_state(), expected_updated_state, 1e-6);
}

TEST(KalmanFilterTest, GetState) {
    // Create a Kalman filter with initial state and covariance.
    VectorXf initial_state(2);
    initial_state << 1, 2;
    MatrixXf initial_covariance(2, 2);
    initial_covariance << 1, 0,
        0, 1;
    KalmanFilter kalman_filter(initial_state, initial_covariance);

    // Assert that the initial state is correct.
    EXPECT_EQ(kalman_filter.get_state(), initial_state);
}

TEST(KalmanFilterTest, GetCovariance) {
    // Create a Kalman filter with initial state and covariance.
    VectorXf initial_state(2);
    initial_state << 1, 2;
    MatrixXf initial_covariance(2, 2);
    initial_covariance << 1, 0,
        0, 1;
    KalmanFilter kalman_filter(initial_state, initial_covariance);

    // Assert that the initial covariance is correct.
    EXPECT_EQ(kalman_filter.get_covariance(), initial_covariance);
}
