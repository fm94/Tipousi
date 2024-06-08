#pragma once

#include <Eigen/Dense>

namespace Tipousi
{
    namespace Optimizer
    {
        class OptimizerBase
        {
          public:
            OptimizerBase(float learning_rate) : m_learning_rate(learning_rate)
            {
            }

            virtual OptimizerBase *clone() const = 0;

            virtual void update_weights(Eigen::MatrixXf &weights,
                                        Eigen::MatrixXf &grads) = 0;

          protected:
            float m_learning_rate;
        };
    };  // namespace Optimizer
};      // namespace Tipousi