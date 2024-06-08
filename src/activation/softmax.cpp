#include "activation/softmax.hpp"

namespace Tipousi
{
    namespace Activation
    {
        void Softmax::forward(const Eigen::MatrixXf &in, Eigen::MatrixXf &out)
        {
            Eigen::MatrixXf expX = in.array().exp();
            out                  = in;
            for (int row = 0; row < in.rows(); ++row)
            {
                out.row(row) = expX.row(row) / expX.row(row).sum();
            }
            m_current_outputs = out;  // cache
        }

        void Softmax::backward(const Eigen::MatrixXf &out_grad,
                               Eigen::MatrixXf       &in_grad)
        {
            in_grad.setZero(out_grad.rows(), out_grad.cols());
            for (int i = 0; i < out_grad.rows(); ++i)
            {
                for (int j = 0; j < out_grad.cols(); ++j)
                {
                    for (int k = 0; k < out_grad.cols(); ++k)
                    {
                        if (j == k)
                        {
                            in_grad(i, j) += out_grad(i, k) *
                                             m_current_outputs(i, k) *
                                             (1.f - m_current_outputs(i, j));
                        }
                        else
                        {
                            in_grad(i, j) += out_grad(i, k) *
                                             m_current_outputs(i, k) *
                                             (-m_current_outputs(i, j));
                        }
                    }
                }
            }
        }
    };  // namespace Activation
};      // namespace Tipousi
