#include "loss/mse.hpp"

namespace Tipousi
{
    namespace Loss
    {
        float MSE::compute(const Eigen::MatrixXf &y,
                           const Eigen::MatrixXf &y_pred) const
        {
            return (y - y_pred).array().square().mean();
        }

        void MSE::grad(Eigen::MatrixXf &out_grad, const Eigen::MatrixXf &y,
                       const Eigen::MatrixXf &y_pred) const
        {
            out_grad = 2.0f * (y_pred - y) / y.rows();
        }
    }  // namespace Loss
}  // namespace Tipousi