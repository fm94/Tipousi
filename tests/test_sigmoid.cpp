#include "activation/sigmoid.hpp"
#include <gtest/gtest.h>

using namespace Tipousi;
using namespace Activation;

TEST(SigmoidLayerTest, ForwardPass)
{
    Eigen::MatrixXf in(2, 3);
    in << 1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f;

    Eigen::MatrixXf out;
    Sigmoid         sigmoid;
    sigmoid.forward(in, out);

    Eigen::MatrixXf expected(2, 3);
    expected << 0.731059f, 0.880797f, 0.952574f, 0.268941f, 0.119203f,
        0.0474259f;

    ASSERT_TRUE(out.isApprox(expected, 1e-5f));
}

TEST(SigmoidLayerTest, BackwardPass)
{
    Eigen::MatrixXf in(2, 3);
    in << 1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f;

    Eigen::MatrixXf dout(2, 3);
    dout << 0.1f, 0.2f, 0.7f, -0.1f, -0.2f, -0.7f;

    Eigen::MatrixXf out;
    Sigmoid         sigmoid;
    sigmoid.forward(in, out);

    Eigen::MatrixXf ddout;
    sigmoid.backward(dout, ddout);

    Eigen::MatrixXf expected(2, 3);
    expected << 0.019661f, 0.020998f, 0.031623f, -0.019661f, -0.020998f,
        -0.031623f;

    ASSERT_TRUE(ddout.isApprox(expected, 1e-4f));
}