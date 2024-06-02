#include "layer/dense.hpp"
#include "utils.hpp"
#include <gtest/gtest.h>

using namespace Tipousi;
using namespace Layer;

TEST(DenseLayerTest, ForwardPass)
{
    Dense dense(3, 2);
    dense.set_learning_rate(0.01f);

    Eigen::MatrixXf input(1, 3);
    input << 1.0f, 2.0f, 3.0f;
    Eigen::MatrixXf output(1, 2);

    dense.forward(input, output);

    Eigen::MatrixXf expected_output = input * dense.get_weights();
    expected_output.rowwise() += dense.get_bias().row(0);
    expectEigenMatrixNear(output, expected_output);
}

TEST(DenseLayerTest, BackwardPass)
{
    Dense dense(3, 2);
    dense.set_learning_rate(0.01f);

    Eigen::MatrixXf input(1, 3);
    input << 1.0f, 2.0f, 3.0f;
    Eigen::MatrixXf output(1, 2);

    dense.forward(input, output);

    Eigen::MatrixXf dout(1, 2);
    dout << 0.1f, 0.2f;
    Eigen::MatrixXf din(1, 3);
    dense.backward(dout, din);

    Eigen::MatrixXf expected_din = dout * dense.get_weights().transpose();
    expectEigenMatrixNear(din, expected_din);
}