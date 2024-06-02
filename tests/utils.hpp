#pragma once

#include <Eigen/Dense>
#include <gtest/gtest.h>

// helper function to compare Eigen matrices
void expectEigenMatrixNear(const Eigen::MatrixXf &a, const Eigen::MatrixXf &b,
                           float tol = 1e-5);
