#pragma once

#include "optimizer/base.hpp"

namespace Tipousi
{
    namespace Optimizer
    {
        class SGD : public OptimizerBase
        {
          public:
            SGD(const float learning_rate);
            ~SGD() = default;

            void update_weights(Eigen::MatrixXf &weights,
                                Eigen::MatrixXf &grads) override;
        };
    };  // namespace Optimizer
};      // namespace Tipousi