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

        virtual void forward(const Eigen::MatrixXf &in, Eigen::MatrixXf &out) = 0;

        virtual void backward(const Eigen::MatrixXf &dout, Eigen::MatrixXf &ddout) = 0;
    };
}; // namespace Tipousi