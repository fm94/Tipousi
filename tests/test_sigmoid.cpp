#include "activation/sigmoid.hpp"
#include <gtest/gtest.h>

using namespace Tipousi;
using namespace Activation;

TEST(SigmoidLayerTest, ForwardPass)
{
    Eigen::MatrixXf in(2, 3);
    in << 1.0, 2.0, 3.0, -1.0, -2.0, -3.0;

    Eigen::MatrixXf out;
    Sigmoid         sigmoid;
    sigmoid.forward(in, out);

    Eigen::MatrixXf expected(2, 3);
    expected << 0.731059, 0.880797, 0.952574, 0.268941, 0.119203, 0.0474259;

    ASSERT_TRUE(out.isApprox(expected, 1e-5));
}

TEST(SigmoidLayerTest, BackwardPass)
{
    Eigen::MatrixXf in(2, 3);
    in << 1.0, 2.0, 3.0, -1.0, -2.0, -3.0;

    Eigen::MatrixXf dout(2, 3);
    dout << 0.1, 0.2, 0.7, -0.1, -0.2, -0.7;

    Eigen::MatrixXf out;
    Sigmoid         sigmoid;
    sigmoid.forward(in, out);

    Eigen::MatrixXf ddout;
    sigmoid.backward(dout, ddout);

    Eigen::MatrixXf expected(2, 3);
    expected << 0.019661, 0.020998, 0.031623, -0.019661, -0.020998, -0.031623;

    ASSERT_TRUE(ddout.isApprox(expected, 1e-4));
}