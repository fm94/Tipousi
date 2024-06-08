#include "optimizer/adam.hpp"
#include "iostream"

namespace Tipousi
{
    namespace Optimizer
    {
        Adam::Adam(float learning_rate, float beta1, float beta2, float epsilon)
            : OptimizerBase(learning_rate), m_beta1(beta1), m_beta2(beta2),
              m_epsilon(epsilon), m_t(0)
        {
        }

        void Adam::update_weights(Eigen::MatrixXf &weights,
                                  Eigen::MatrixXf &grads)
        {
            if (m_m.size() == 0)
            {
                m_m = Eigen::MatrixXf::Zero(grads.rows(), grads.cols());
                m_v = Eigen::MatrixXf::Zero(grads.rows(), grads.cols());
            }

            m_t++;

            // update biased first moment estimate
            m_m = m_beta1 * m_m + (1.0 - m_beta1) * grads;

            // update biased second raw moment estimate
            m_v = m_beta2 * m_v +
                  (1.0 - m_beta2) * grads.array().square().matrix();

            double bias_correction1 = 1.0 - std::pow(m_beta1, m_t) + m_epsilon;
            double bias_correction2 = 1.0 - std::pow(m_beta2, m_t) + m_epsilon;

            // compute bias-corrected first moment estimate
            Eigen::MatrixXf m_hat = m_m.array() / bias_correction1;

            // compute bias-corrected second raw moment estimate
            Eigen::MatrixXf v_hat = m_v.array() / bias_correction2;

            // update weights
            weights.array() -= m_learning_rate * m_hat.array() /
                               (v_hat.array().sqrt() + m_epsilon);
        }
    }  // namespace Optimizer
}  // namespace Tipousi