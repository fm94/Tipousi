#include "activation/softmax.hpp"

namespace Tipousi
{
    namespace Activation
    {
        void Softmax::forward(const Eigen::MatrixXf &in, Eigen::MatrixXf &out)
        {
            // compute the maximum value per row for numerical stability
            Eigen::VectorXf row_max    = in.rowwise().maxCoeff();
            Eigen::MatrixXf shifted_in = in.colwise() - row_max;
            Eigen::MatrixXf expX       = shifted_in.array().exp();
            Eigen::VectorXf sumExpX    = expX.rowwise().sum();
            out = expX.array().colwise() / sumExpX.array();

            // here we cache outputs instead of inputs
            m_current_outputs = out;
        }

        void Softmax::backward(const Eigen::MatrixXf &out_grad,
                               Eigen::MatrixXf       &in_grad)
        {
            // initialize ddout with the correct size
            in_grad.resizeLike(out_grad);

            // compute gradient of softmax
            for (int i = 0; i < out_grad.rows(); ++i)
            {
                Eigen::MatrixXf diag  = m_current_outputs.row(i).asDiagonal();
                Eigen::MatrixXf outer = m_current_outputs.row(i).transpose() *
                                        m_current_outputs.row(i);
                Eigen::MatrixXf jacobian = diag - outer;

                // compute in_grad as the matrix-vector product
                in_grad.row(i) = out_grad.row(i) * jacobian;
            }
        }
    };  // namespace Activation
};      // namespace Tipousi
