#include "activation/softmax.hpp"
#include "utils.hpp"
#include <gtest/gtest.h>

using namespace Tipousi;
using namespace Activation;

TEST(SoftmaxLayerTest, ForwardPass)
{
    Softmax         softmax;
    Eigen::MatrixXf input(2, 2);
    input << 1.0f, 2.0f, 3.0f, 4.0f;

    Eigen::MatrixXf output;
    softmax.forward(input, output);

    Eigen::MatrixXf expected_output(2, 2);
    expected_output << 0.268941f, 0.731059f, 0.268941f, 0.731059f;

    expectEigenMatrixNear(output, expected_output, 1e-5f);
}

TEST(SoftmaxLayerTest, BackwardPass)
{
    Softmax         softmax;
    Eigen::MatrixXf input(2, 2);
    input << 1.0f, 2.0f, 3.0f, 4.0f;

    Eigen::MatrixXf output;
    softmax.forward(input, output);

    Eigen::MatrixXf out_grad(2, 2);
    out_grad << 0.1f, -0.1f, 0.1f, -0.1f;

    Eigen::MatrixXf in_grad;
    softmax.backward(out_grad, in_grad);

    // manually calculated
    Eigen::MatrixXf expected_in_grad(2, 2);
    expected_in_grad << 0.0393f, -0.0393f, 0.0393f, -0.0393f;

    expectEigenMatrixNear(in_grad, expected_in_grad, 1e-3f);
}