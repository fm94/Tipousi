#pragma once

#include "loss/base.hpp"

namespace Tipousi
{
    namespace Loss
    {
        class MSE : public LossBase
        {
          public:
            MSE()  = default;
            ~MSE() = default;

            float compute(const Eigen::MatrixXf &y,
                          const Eigen::MatrixXf &y_pred) const override;

            void grad(Eigen::MatrixXf &out_grad, const Eigen::MatrixXf &y,
                      const Eigen::MatrixXf &y_pred) const override;
        };
    };  // namespace Loss
};      // namespace Tipousi