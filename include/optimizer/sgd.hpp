#pragma once

#include "optimizer/base.hpp"

namespace Tipousi
{
    namespace Optimizer
    {
        class SGD : public OptimizerBase
        {
          public:
            SGD(float learning_rate);
            ~SGD() = default;

            void update_weights(Eigen::MatrixXf &weights,
                                Eigen::MatrixXf &grads) override;

            virtual SGD *clone() const override { return new SGD(*this); }
        };
    };  // namespace Optimizer
};      // namespace Tipousi