#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {
}

KalmanFilter::~KalmanFilter() {
}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
    MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  // regular Kalman Filter update step

  VectorXd y = z - (H_ * x_);

  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd KalmanGain = P_ * Ht * Si; //1D equiv=> g = p/p+r  = p * (p+r)^-1

  x_ = x_ + (KalmanGain * y); // x_pred+ g * ( z - x_pred)  where g - Kalman Gain

  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - KalmanGain * H_) * P_; // 1D equiv => p = (1-g) * p
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {

  double px = x_(0);
  double py = x_(1);
  double vx = x_(2);
  double vy = x_(3);

  float c1 = px * px + py * py;

  if (fabs(c1) < 0.001) {
    std::cout << "ERROR: px and py both cannot be zero. Setting c1 to 0.001"<< std::endl;
    c1 = 0.001;
  }

  float c2 = sqrt(c1); // sqrt(px^2 + py^2 )

  VectorXd h = VectorXd(3); // the non linear function h (will hold polar coordinates of x_pred)
  h[0] = c2;
  h[1] = atan2(py, px);
  h[2] = (px * vx + py * vy) / c2;

  // extended KF z is in polar coordinates. so error y should be computed from h
  VectorXd y = z - h;

  // after we subtract h from z, the resultant y vector can have a phi out of -pi to +pi range
  // we need to normalize it
  y[1] = atan2(sin((double) y[1]), cos((double) y[1]));

  // note : H_ is already set to Jacobian from the calling program.
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_; // R_ should be set by caller to R_radar
  MatrixXd Si = S.inverse();
  MatrixXd KalmanGain = P_ * Ht * Si; //1D equiv=> g = p/p+r  = p * (p+r)^-1

  x_ = x_ + (KalmanGain * y); // x_pred+ g * ( z - x_pred)  where g - Kalman Gain

  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - KalmanGain * H_) * P_; // 1D equiv => p = (1-g) * p

}
