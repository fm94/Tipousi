#include "layer/dense.hpp"
#include <iostream>

namespace Tipousi
{
    namespace Layer
    {
        Dense::Dense(int in_size, int out_size)
        {
            m_weights = Eigen::MatrixXf::Random(in_size, out_size);
            m_bias    = Eigen::MatrixXf::Random(1, out_size);
        }

        void Dense::forward(const Eigen::MatrixXf &in, Eigen::MatrixXf &out)
        {
            m_current_inputs = in;  // cache
            out              = in * m_weights;
            out.rowwise() += m_bias.row(0);
        }

        void Dense::backward(const Eigen::MatrixXf &out_grad,
                             Eigen::MatrixXf       &in_grad)
        {
            Eigen::MatrixXf weight_grad =
                m_current_inputs.transpose() * out_grad;
            Eigen::MatrixXf bias_grad = out_grad.colwise().sum();
            m_optimizer->update_weights(m_weights, weight_grad);
            // .row(0) has been removed here! check whether it has "no" effect
            m_optimizer->update_weights(m_bias, bias_grad);
            in_grad = out_grad * m_weights.transpose();
        }
    }  // namespace Layer
}  // namespace Tipousi