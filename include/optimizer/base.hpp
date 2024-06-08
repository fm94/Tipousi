#pragma once

#include <Eigen/Dense>

namespace Tipousi
{
    namespace Optimizer
    {
        class OptimizerBase
        {
          public:
            virtual void update_weights(Eigen::MatrixXf &weights,
                                        Eigen::MatrixXf &grads) = 0;

          protected:
            float m_learning_rate;
        };
    };  // namespace Optimizer
};      // namespace Tipousi