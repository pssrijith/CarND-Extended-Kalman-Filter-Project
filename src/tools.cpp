#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {
}

Tools::~Tools() {
}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
    const vector<VectorXd> &ground_truth) {
  VectorXd rmse = VectorXd(4);
  rmse << 0, 0, 0, 0; // px, py, vx, vy
  int est_size = estimations.size();

  if (est_size == 0) {
    cout << "Error: Estimations vector cannot be a 0 length vector" << endl;
    return rmse;
  }

  if (est_size != ground_truth.size()) {
    cout << "Error: Estimations and ground_truth vectors must have same length"
        << endl;
    return rmse;
  }

  for (int i = 0; i < est_size; i++) {
    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array() * residual.array(); // elementwise multiplication
    rmse += residual;
  }

  // mean rmse
  rmse = rmse / est_size;

  // root mean rmse for each of (px, py, vx, vy)
  rmse = rmse.array().sqrt();

  return rmse;

}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {

  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  // helpful terms for computing Jacobian
  float c1 = px * px + py * py; // px^2 + py^2
  float c2 = sqrt(c1); // sqrt(px^2 + py^2)
  float c3 = c1 * c2;

  MatrixXd Hj(3, 4);

  if (fabs(c1) < 0.001) {
    c1 = 0.001;
  }

  if (fabs(c3) < 0.001) {
    c3 = 0.001;
  }

  Hj <<   px / c2,                  py / c2,                          0, 0,
         -py / c1,                  px / c1,                          0, 0,
     py* (vx * py - vy * px) / c3,   px * (vy * px - vx * py) / c3,  px / c2,  py/ c2;

  return Hj;
}
