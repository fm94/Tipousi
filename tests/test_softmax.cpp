#include "activation/softmax.hpp"
#include <gtest/gtest.h>

using namespace Tipousi;
using namespace Activation;

TEST(SoftmaxLayerTest, ForwardPass)
{
    Eigen::MatrixXf in(2, 3);
    in << 1.0, 2.0, 3.0, 1.0, 2.0, -1.0;

    Eigen::MatrixXf out;
    Softmax         softmax;
    softmax.forward(in, out);

    Eigen::MatrixXf expected(2, 3);
    expected << 0.0900306, 0.244728, 0.665241, 0.259496, 0.705385, 0.035119;

    ASSERT_TRUE(out.isApprox(expected, 1e-5));
}

TEST(SoftmaxLayerTest, BackwardPass)
{
    Eigen::MatrixXf in(2, 3);
    in << 1.0, 2.0, 3.0, 1.0, 2.0, -1.0;

    Eigen::MatrixXf dout(2, 3);
    dout << 0.1, 0.2, 0.7, 0.3, 0.4, 0.3;

    Eigen::MatrixXf out;
    Softmax         softmax;
    softmax.forward(in, out);

    Eigen::MatrixXf ddout;
    softmax.backward(dout, ddout);

    // manually computed expected ddout values
    Eigen::MatrixXf expected(2, 3);
    expected << -0.038138, -0.079198, 0.117337, -0.018304, 0.020781, -0.002477;

    ASSERT_TRUE(ddout.isApprox(expected, 1e-4));
}