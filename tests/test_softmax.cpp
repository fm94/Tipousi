#include "activation/softmax.hpp"
#include <gtest/gtest.h>

using namespace Tipousi;
using namespace Activation;

TEST(SoftmaxLayerTest, ForwardPass)
{
    Eigen::MatrixXf in(2, 3);
    in << 1.0f, 2.0f, 3.0f, 1.0f, 2.0f, -1.0f;

    Eigen::MatrixXf out;
    Softmax         softmax;
    softmax.forward(in, out);

    Eigen::MatrixXf expected(2, 3);
    expected << 0.0900306f, 0.244728f, 0.665241f, 0.259496f, 0.705385f,
        0.035119f;

    ASSERT_TRUE(out.isApprox(expected, 1e-5f));
}

TEST(SoftmaxLayerTest, BackwardPass)
{
    Eigen::MatrixXf in(2, 3);
    in << 1.0f, 2.0f, 3.0f, 1.0f, 2.0f, -1.0f;

    Eigen::MatrixXf dout(2, 3);
    dout << 0.1f, 0.2f, 0.7f, 0.3f, 0.4f, 0.3f;

    Eigen::MatrixXf out;
    Softmax         softmax;
    softmax.forward(in, out);

    Eigen::MatrixXf ddout;
    softmax.backward(dout, ddout);

    // manually computed expected ddout values
    Eigen::MatrixXf expected(2, 3);
    expected << -0.038138f, -0.079198f, 0.117337f, -0.018304f, 0.020781f,
        -0.002477f;

    ASSERT_TRUE(ddout.isApprox(expected, 1e-4f));
}