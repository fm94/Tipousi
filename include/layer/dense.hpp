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

            void backward(const Eigen::MatrixXf &out_grad,
                          Eigen::MatrixXf       &in_grad) override;

            Eigen::MatrixXf get_weights() { return m_weights; }
            Eigen::MatrixXf get_bias() { return m_bias; }

          private:
            Eigen::MatrixXf m_weights;
            Eigen::MatrixXf m_bias;
        };
    };  // namespace Layer
};      // namespace Tipousi