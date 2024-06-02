#include "activation/softmax.hpp"
#include "utils.hpp"
#include <gtest/gtest.h>

using namespace Tipousi;
using namespace Activation;

TEST(SoftmaxLayerTest, ForwardPass)
{
    Softmax         softmax;
    Eigen::MatrixXf input(2, 2);
    input << 1.0, 2.0, 3.0, 4.0;

    Eigen::MatrixXf output;
    softmax.forward(input, output);

    Eigen::MatrixXf expected_output(2, 2);
    expected_output << 0.268941, 0.731059, 0.268941, 0.731059;

    expectEigenMatrixNear(output, expected_output, 1e-5);
}

TEST(SoftmaxLayerTest, BackwardPass)
{
    Softmax         softmax;
    Eigen::MatrixXf input(2, 2);
    input << 1.0, 2.0, 3.0, 4.0;

    Eigen::MatrixXf output;
    softmax.forward(input, output);

    Eigen::MatrixXf out_grad(2, 2);
    out_grad << 0.1, -0.1, 0.1, -0.1;

    Eigen::MatrixXf in_grad;
    softmax.backward(out_grad, in_grad);

    // manually calculated
    Eigen::MatrixXf expected_in_grad(2, 2);
    expected_in_grad << 0.0393, -0.0393, 0.0393, -0.0393;

    expectEigenMatrixNear(in_grad, expected_in_grad, 1e-3);
}