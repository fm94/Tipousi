#include "utils.hpp"

// helper function to compare Eigen matrices
void expectEigenMatrixNear(const Eigen::MatrixXf &a, const Eigen::MatrixXf &b,
                           float tol)
{
    ASSERT_EQ(a.rows(), b.rows());
    ASSERT_EQ(a.cols(), b.cols());
    for (int i = 0; i < a.rows(); ++i)
    {
        for (int j = 0; j < a.cols(); ++j)
        {
            EXPECT_NEAR(a(i, j), b(i, j), tol);
        }
    }
}
