#include "activation/softmax.hpp"

namespace Tipousi
{
    namespace Activation
    {
        Softmax::Softmax() {}

        void Softmax::forward(const Eigen::MatrixXf &in, Eigen::MatrixXf &out)
        {
            Eigen::MatrixXf expX    = in.array().exp();
            Eigen::VectorXf sumExpX = expX.rowwise().sum();
            out =
                (expX.array().rowwise() / sumExpX.transpose().array()).matrix();
        }

        void Softmax::backward(const Eigen::MatrixXf &dout,
                               Eigen::MatrixXf       &ddout)
        {
        }
    };  // namespace Activation
};      // namespace Tipousi
