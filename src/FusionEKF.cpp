#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  // state transition matrix
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1, 0, 1, 0,
             0, 1, 0, 1,
             0, 0, 1, 0,
             0, 0, 0, 1;

  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  // initialize acceleration noise components
  noise_ax = 9.0;
  noise_ay = 9.0;

}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {
}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {

    /*
     * Initializing the state ekf_.x_ with the first measurement.
     */

    // we compute/get the px and py measurements from RADAR/LIDAR. vx and vy are initialized to 0
    float px, py, vx = 0.0, vy = 0.0;
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
       * Convert radar from polar to cartesian coordinates.
       * */
      float rho = measurement_pack.raw_measurements_[0];
      float phi = measurement_pack.raw_measurements_[1];
      float rho_dot = measurement_pack.raw_measurements_[2];

      px = rho * cos(phi);
      py = rho * sin(phi);
      vx = rho_dot * cos(phi);
      vy = rho_dot * sin(phi);

    } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {

      px = measurement_pack.raw_measurements_[0];
      py = measurement_pack.raw_measurements_[1];

    }

    ekf_.x_ = VectorXd(4);
    ekf_.x_ << px, py, vx, vy;

    // state co-variance matrix
    ekf_.P_ = MatrixXd(4, 4);
    ekf_.P_ << 1, 0, 0, 0,
               0, 1, 0, 0,
               0, 0, 1000, 0,
               0, 0, 0, 1000;

    previous_timestamp_ = measurement_pack.timestamp_;
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
  //dt - expressed in seconds
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;

  // Update the state Transition matrix F_ with dt
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  float dt2 = dt * dt;
  float dt3 = dt2 * dt;
  float dt4 = dt3 * dt;

  // process noise correlation matrix
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ << dt4/4 * noise_ax, 0, dt3/2 * noise_ax, 0,
            0, dt4/4* noise_ay, 0, dt3/2 * noise_ay,
            dt3/2 * noise_ax, 0, dt2 * noise_ax, 0,
            0, dt3/2 * noise_ay, 0, dt2 * noise_ay;

  // predict x = F * x ; P = F * P * F.transpose() + Q
  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    // calculate Jacobian of predicted state
    Hj_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.H_ = Hj_, ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);

  } else {

    // Laser updates. use H_laser and R_laser
    ekf_.H_ = H_laser_, ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  //cout << "x_ = " << ekf_.x_ << endl;
  //cout << "P_ = " << ekf_.P_ << endl;
}
