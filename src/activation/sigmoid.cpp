#include "activation/sigmoid.hpp"

namespace Tipousi
{
    namespace Activation
    {
        void Sigmoid::forward(const Eigen::MatrixXf &in, Eigen::MatrixXf &out)
        {
            out               = 1.0 / (1.0 + (-in.array()).exp());
            m_current_outputs = out;  // cache
        }

        void Sigmoid::backward(const Eigen::MatrixXf &out_grad,
                               Eigen::MatrixXf       &in_grad)
        {
            const Eigen::MatrixXf &y = m_current_outputs;
            in_grad = out_grad.array() * y.array() * (1.0 - y.array());
        }
    };  // namespace Activation
};      // namespace Tipousi
