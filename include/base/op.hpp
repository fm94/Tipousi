/**
 * Base class for all kinds of layers
 */

#pragma once

#include "optimizer/base.hpp"
#include <Eigen/Dense>
#include <memory>

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

        virtual void set_optimizer(Optimizer::OptimizerBase &optimizer)
        {
            m_optimizers.emplace_back(optimizer.clone());
        };

        virtual int get_n_trainable_params() { return 0; }

      protected:
        Eigen::MatrixXf m_current_inputs;
        Eigen::MatrixXf m_current_outputs;

        std::vector<std::unique_ptr<Optimizer::OptimizerBase>> m_optimizers{};
    };
};  // namespace Tipousi