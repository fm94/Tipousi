#include "layer/dense.hpp"

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
            m_current_inputs = in;           // caching
            out.noalias() = in * m_weights;  // efficient matrix multiplication
            out.rowwise() += m_bias.row(0);
        }

        void Dense::backward(const Eigen::MatrixXf &out_grad,
                             Eigen::MatrixXf       &in_grad)
        {
            // gradient with respect to weights and bias
            Eigen::MatrixXf weight_grad =
                m_current_inputs.transpose() * out_grad;
            Eigen::MatrixXf bias_grad = out_grad.colwise().sum();

            // update weights and biases
            m_weights.noalias() -= m_learning_rate * weight_grad;
            m_bias.row(0).noalias() -= m_learning_rate * bias_grad;

            // gradient with respect to input
            in_grad.noalias() = out_grad * m_weights.transpose();
        }
    }  // namespace Layer
}  // namespace Tipousi