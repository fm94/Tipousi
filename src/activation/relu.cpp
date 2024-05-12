#include "activation/relu.hpp"

namespace Tipousi
{
    namespace Activation
    {
        ReLU::ReLU()
        {
        }

        void ReLU::forward(const Eigen::MatrixXf &in, Eigen::MatrixXf &out)
        {
            out = (in.array().max(0)).matrix();
        }

        void ReLU::backward(const Eigen::MatrixXf &dout, Eigen::MatrixXf &ddout)
        {
        }
    };
};
