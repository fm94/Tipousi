#include "optimizer/sgd.hpp"

namespace Tipousi
{
    namespace Optimizer
    {
        SGD::SGD(float learning_rate) : OptimizerBase(learning_rate) {}

        void SGD::update_weights(Eigen::MatrixXf &weights,
                                 Eigen::MatrixXf &grads)
        {
            weights -= m_learning_rate * grads;
        }
    }  // namespace Optimizer
}  // namespace Tipousi