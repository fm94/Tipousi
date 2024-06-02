#include "activation/relu.hpp"

namespace Tipousi
{
    namespace Activation
    {
        void ReLU::forward(const Eigen::MatrixXf &in, Eigen::MatrixXf &out)
        {
            m_current_inputs = in;  // cache
            out              = in.array().max(0);
        }

        void ReLU::backward(const Eigen::MatrixXf &out_grad,
                            Eigen::MatrixXf       &in_grad)
        {
            in_grad =
                out_grad.array() * (m_current_inputs.array() > 0).cast<float>();
        }
    };  // namespace Activation
};      // namespace Tipousi
