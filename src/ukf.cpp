#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {

  initialized_ = false;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // in us
  time_us_ = 0;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.8;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.55;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // state dimension (x, y, velocity, yaw_angle, yaw_rate)
  n_x_ = 5;

  // augmented state (n_x_ + 2)
  n_aug_ = 7;

  // hyper param for sigma points, 3 - n_aug_
  lambda_ = -4;

  // set weights
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  weights_.tail(2 * n_aug_).fill(0.5 / (lambda_ + n_aug_));

  // initial state vector
  x_ = VectorXd(n_x_);
  x_.fill(0);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  // matrix for sigma points
  Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // matrix for predicted sigma points
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
}

UKF::~UKF() {}

void UKF::Init(MeasurementPackage meas_package){

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR){
    double rho = meas_package.raw_measurements_[0];
    double phi = meas_package.raw_measurements_[1];
    double px = cos(phi) * rho;
    double py = sin(phi) * rho;
    x_ << px, py, 0, 0, 0;
  } else if  (meas_package.sensor_type_ == MeasurementPackage::LASER){
    double px = meas_package.raw_measurements_[0];
    double py = meas_package.raw_measurements_[1];
    x_ << px, py, 0, 0, 0;
  } else{
    return;
  }

  time_us_ = meas_package.timestamp_;

  P_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 1;

  // set Initialize flag to true
  initialized_ = true;
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  // Initial measurement
  if (!initialized_) {
    Init(meas_package);
    return;
  }

  // Record timestamp
  double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0; // delta_t in seconds
  time_us_ = meas_package.timestamp_;

  // Predict
  while (delta_t > 0.1){
    const double step = 0.05;
    Prediction(step);
    delta_t -= step;
  }
  Prediction(delta_t);

  // Update
  if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
  } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  }
}

void UKF::AugmentSigmaPoints() {
  //create augmented mean state
  VectorXd x_aug_ = VectorXd(n_aug_);
  x_aug_.fill(0);
  x_aug_.head(5) = x_;

  //create augmented covariance matrix
  MatrixXd P_aug_ = MatrixXd(n_aug_, n_aug_);
  P_aug_.fill(0.0);
  P_aug_.topLeftCorner(n_x_, n_x_) = P_;
  P_aug_(n_aug_ - 2, n_aug_ - 2) = std_a_ * std_a_;
  P_aug_(n_aug_ - 1, n_aug_ - 1) = std_yawdd_ * std_yawdd_;

  //create augmented sigma points
  MatrixXd A = P_aug_.llt().matrixL();
  Xsig_aug_.col(0) = x_aug_;
  for (int i = 0; i < n_aug_; i++) {
    Xsig_aug_.col(i + 1)          = x_aug_ + sqrt(lambda_ + n_aug_) * A.col(i);
    Xsig_aug_.col(i + 1 + n_aug_) = x_aug_ - sqrt(lambda_ + n_aug_) * A.col(i);
  }
}

void UKF::PredictSigmaPoints(double delta_t) {
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    double px = Xsig_aug_(0,i);
    double py = Xsig_aug_(1,i);
    double v = Xsig_aug_(2,i);
    double yaw = Xsig_aug_(3,i);
    double yawd = Xsig_aug_(4,i);
    double nu_a = Xsig_aug_(5,i);
    double nu_yawdd = Xsig_aug_(6,i);

    //predicted state values
    double px_pred, py_pred, v_pred, yaw_pred, yawd_pred;

    if (fabs(yawd) > 0.001) {
      px_pred = px + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
      py_pred = py + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
    } else { // avoid divide by zero
      px_pred = px + v * delta_t * cos(yaw);
      py_pred = py + v * delta_t * sin(yaw);
    }

    //add noise
    px_pred += 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_pred += 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_pred = v + nu_a * delta_t;

    yaw_pred = yaw + yawd * delta_t + 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_pred = yawd + nu_yawdd * delta_t;

    Xsig_pred_(0, i) = px_pred;
    Xsig_pred_(1, i) = py_pred;
    Xsig_pred_(2, i) = v_pred;
    Xsig_pred_(3, i) = yaw_pred;
    Xsig_pred_(4, i) = yawd_pred;
  }
}

void UKF::PredictMeanAndCovariance() {
  // predict mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    x_ += weights_(i) * Xsig_pred_.col(i);
  }
  // predict covariance
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    while (x_diff(3) >  M_PI) x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;
    P_ += weights_(i) * x_diff * x_diff.transpose() ;
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  AugmentSigmaPoints();
  PredictSigmaPoints(delta_t);
  PredictMeanAndCovariance();
}

void UKF::PredictMeasurement(int n_z, const MatrixXd &Zsig, VectorXd &z_pred, MatrixXd &S, MatrixXd &R) {

  // predict measurement mean
  z_pred.fill(0.0);
  for (int i=0; i < 2 * n_aug_ + 1; i++) {
    z_pred += weights_(i) * Zsig.col(i);
  }

  // measurement covariance matrix S
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    while (z_diff(1) >  M_PI) z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;
    S += weights_(i) * z_diff * z_diff.transpose();
  }

  S += R;
}

void UKF::UpdateState(const VectorXd &z, const VectorXd &z_pred, const MatrixXd &S, const MatrixXd &Zsig) {

  int n_z = z_pred.rows();

  // calculate cross correlation
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    while (z_diff(1) >  M_PI) z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    while (x_diff(3) >  M_PI) x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  MatrixXd K = Tc * S.inverse();  // Kalman gain K
  VectorXd z_diff = z - z_pred;

  while (z_diff(1) >  M_PI) z_diff(1) -= 4. * M_PI;
  while (z_diff(1) < -M_PI) z_diff(1) += 4. * M_PI;

  //update
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {

  VectorXd z = meas_package.raw_measurements_;

  MatrixXd Zsig = MatrixXd(2, 2 * n_aug_ + 1);
  VectorXd z_pred = VectorXd(2);
  MatrixXd S = MatrixXd(2, 2);

  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    double px = Xsig_pred_(0, i);
    double py = Xsig_pred_(1, i);
    Zsig(0, i) = px;
    Zsig(1, i) = py;
  }

  // measurement noise covariance matrix
  MatrixXd R = MatrixXd(2, 2);
  R << std_laspx_ * std_laspx_, 0,
       0, std_laspy_ * std_laspy_;

  // Predict radar measurement
  PredictMeasurement(2, Zsig, z_pred, S, R);
  // Update state
  UpdateState(z, z_pred, S, Zsig);
  // Calculate NIS
  NIS_laser_ = (z - z_pred).transpose() * S.inverse() * (z - z_pred);
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {

  VectorXd z = meas_package.raw_measurements_;

  MatrixXd Zsig = MatrixXd(3, 2 * n_aug_ + 1);
  VectorXd z_pred = VectorXd(3);
  MatrixXd S = MatrixXd(3, 3);

  //transform sigma points into radar measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    double px = Xsig_pred_(0, i);
    double py = Xsig_pred_(1, i);
    double v   = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);
    double vx = cos(yaw) * v;
    double vy = sin(yaw) * v;
    Zsig(0, i) = sqrt(px * px + py * py); //rho
    Zsig(1, i) = atan2(py, px); //phi
    Zsig(2, i) = (px * vx + py * vy ) / sqrt(px * px + py * py); //rho_dot
  }

  //measurement noise covariance matrix
  MatrixXd R = MatrixXd(3, 3);
  R << std_radr_ * std_radr_, 0, 0,
       0, std_radphi_ * std_radphi_, 0,
       0, 0, std_radrd_ * std_radrd_;

  // Predict radar measurement with given Sigma predictions
  PredictMeasurement(3, Zsig, z_pred, S, R);
  // Update the state
  UpdateState(z, z_pred, S, Zsig);
  // Calculate NIS
  NIS_radar_ = (z - z_pred).transpose() * S.inverse() * (z - z_pred);
}
