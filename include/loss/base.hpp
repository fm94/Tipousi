#pragma once

#include <Eigen/Dense>

namespace Tipousi
{
    namespace Loss
    {
        class LossBase
        {
          public:
            LossBase()          = default;
            virtual ~LossBase() = default;

            virtual float compute(const Eigen::MatrixXf &y,
                                  const Eigen::MatrixXf &y_pred) const = 0;

            virtual void grad(Eigen::MatrixXf       &out_grad,
                              const Eigen::MatrixXf &y,
                              const Eigen::MatrixXf &y_pred) const = 0;
        };
    };  // namespace Loss
};      // namespace Tipousi