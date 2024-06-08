#pragma once

#include "base/op.hpp"

namespace Tipousi
{
    namespace Activation
    {
        class Sigmoid : public Op
        {
          public:
            Sigmoid()           = default;
            ~Sigmoid() override = default;

            void forward(const Eigen::MatrixXf &in,
                         Eigen::MatrixXf       &out) override;

            void backward(const Eigen::MatrixXf &out_grad,
                          Eigen::MatrixXf       &in_grad) override;
        };
    };  // namespace Activation
};      // namespace Tipousi