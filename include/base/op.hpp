/**
 * Base class for all kinds of layers
 */

#pragma once

#include <Eigen/Dense>

namespace Tipousi
{
    class Op
    {
      public:
        virtual ~Op() = default;

        virtual void forward(const Eigen::MatrixXf &in,
                             Eigen::MatrixXf       &out) = 0;

        virtual void backward(const Eigen::MatrixXf &out_grad,
                              Eigen::MatrixXf       &in_grad) = 0;

        void set_learning_rate(float learning_rate)
        {
            m_learning_rate = learning_rate;
        }

      protected:
        float           m_learning_rate;
        Eigen::MatrixXf m_current_inputs;
    };
};  // namespace Tipousi