#pragma once

#include "base/op.hpp"

namespace Tipousi
{
    namespace Layer
    {
        class Dense : public Op
        {
          public:
            Dense(int in_size, int out_size);
            ~Dense() = default;

            void forward(const Eigen::MatrixXf &in,
                         Eigen::MatrixXf       &out) override;

            void backward(const Eigen::MatrixXf &dout,
                          Eigen::MatrixXf       &ddout) override;

          private:
            Eigen::MatrixXf m_weights;
            Eigen::MatrixXf m_bias;
        };
    };  // namespace Layer
};      // namespace Tipousi