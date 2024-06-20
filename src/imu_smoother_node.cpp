#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <tf2_eigen/tf2_eigen.h>
#include <eigen3/Eigen/Eigen>
#include <eigen_conversions/eigen_msg.h>
#include <std_msgs/Header.h>
#include <algorithm>


std::vector<Eigen::Vector3d> g_accel_buffer;
std::vector<Eigen::Vector3d> g_gyro_buffer;
std::vector<sensor_msgs::Imu> g_msg_buffer;
int g_window_size;
ros::Subscriber g_sub;
ros::Publisher g_pub;

void clampVector(Eigen::Vector3d& vec, float min, float max) {
  auto clamp = [min, max](float x) {
    if (x < min) {
      return min;
    }
    if (x > max) {
      return max;
    }
    return x;
  };
  vec.x() = clamp(vec.x());
  vec.y() = clamp(vec.y());
  vec.z() = clamp(vec.z());
}

void imuCallback(const sensor_msgs::Imu& msg)
{
  Eigen::Vector3d accel;
  Eigen::Vector3d gyro;
  tf::vectorMsgToEigen(msg.linear_acceleration, accel);
  tf::vectorMsgToEigen(msg.angular_velocity, gyro);
  clampVector(accel, -10, 10);
  clampVector(gyro, 4, 4);
  g_accel_buffer.push_back(accel);
  g_gyro_buffer.push_back(gyro);
  g_msg_buffer.push_back(msg);
}

void timerCallback(const ros::TimerEvent&)
{
  if (g_accel_buffer.size() < g_window_size) {
    return;
  }
  Eigen::Vector3d accel;
  Eigen::Vector3d gyro;
  for (int i = 0; i < g_window_size; i++) {
    int index = g_accel_buffer.size() - g_window_size + i;
    accel += g_accel_buffer.at(index);
    gyro += g_gyro_buffer.at(index);
  }

  accel /= g_window_size;
  gyro /= g_window_size;

  sensor_msgs::Imu msg;
  tf::vectorEigenToMsg(accel, msg.linear_acceleration);
  tf::vectorEigenToMsg(gyro, msg.angular_velocity);
  auto last_msg = g_msg_buffer.back();
  msg.header = last_msg.header;
  msg.linear_acceleration_covariance = last_msg.linear_acceleration_covariance;
  msg.angular_velocity_covariance = last_msg.angular_velocity_covariance;
  msg.orientation = last_msg.orientation;
  msg.orientation_covariance = last_msg.orientation_covariance;

  g_pub.publish(msg);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "imu_smoother");

  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");
  nh.param("window_size", g_window_size, 3);

  g_sub = nh.subscribe("imu", 1000, imuCallback);
  g_pub = private_nh.advertise<sensor_msgs::Imu>("smoothed_imu", 1);

  ros::Timer timer = nh.createTimer(ros::Duration(0.02), timerCallback);

  ros::spin();

  return 0;
}


