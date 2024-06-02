#include "activation/relu.hpp"
#include "utils.hpp"
#include <gtest/gtest.h>

using namespace Tipousi;
using namespace Activation;

TEST(ReLULayerTest, ForwardPass)
{
    ReLU            relu;
    Eigen::MatrixXf input(1, 3);
    input << 1.0, -2.0, 3.0;
    Eigen::MatrixXf output(1, 3);

    relu.forward(input, output);

    Eigen::MatrixXf expected_output(1, 3);
    expected_output << 1.0, 0.0, 3.0;
    expectEigenMatrixNear(output, expected_output);
}

TEST(ReLULayerTest, BackwardPass)
{
    ReLU            relu;
    Eigen::MatrixXf input(1, 3);
    input << 1.0, -2.0, 3.0;
    Eigen::MatrixXf output(1, 3);

    relu.forward(input, output);

    Eigen::MatrixXf dout(1, 3);
    dout << 0.1, 0.2, 0.3;
    Eigen::MatrixXf din(1, 3);
    relu.backward(dout, din);

    Eigen::MatrixXf expected_din(1, 3);
    expected_din << 0.1, 0.0, 0.3;
    expectEigenMatrixNear(din, expected_din);
}