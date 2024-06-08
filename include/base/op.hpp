/**
 * Base class for all kinds of layers
 */

#pragma once

#include "optimizer/base.hpp"
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

        void set_optimizer(Optimizer::OptimizerBase *optimizer)
        {
            m_optimizer = optimizer;
        };

      protected:
        Eigen::MatrixXf m_current_inputs;
        Eigen::MatrixXf m_current_outputs;

        Optimizer::OptimizerBase *m_optimizer;
    };
};  // namespace Tipousi