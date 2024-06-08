#include "optimizer/sgd.hpp"

namespace Tipousi
{
    namespace Optimizer
    {
        SGD::SGD(const float learning_rate) { m_learning_rate = learning_rate; }

        void Optimizer::SGD::update_weights(Eigen::MatrixXf &weights,
                                            Eigen::MatrixXf &grads)
        {
            weights -= m_learning_rate * grads;
        }
    }  // namespace Optimizer
}  // namespace Tipousi