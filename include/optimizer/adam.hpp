#pragma once

#include "optimizer/base.hpp"

namespace Tipousi
{
    namespace Optimizer
    {
        class Adam : public OptimizerBase
        {
          public:
            Adam(float learning_rate, float beta1 = 0.9f, float beta2 = 0.999f,
                 float epsilon = 1e-8f);
            ~Adam() = default;

            void update_weights(Eigen::MatrixXf &weights,
                                Eigen::MatrixXf &grads) override;

            virtual Adam *clone() const override { return new Adam(*this); }

          private:
            float           m_beta1, m_beta2, m_epsilon;
            Eigen::MatrixXf m_m, m_v;
            int             m_t;
        };
    };  // namespace Optimizer
};      // namespace Tipousi