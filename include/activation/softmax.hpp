#pragma once

#include "base/op.hpp"

namespace Tipousi
{
    namespace Activation
    {
        class Softmax : public Op
        {
            public:
                Softmax();
                ~Softmax() override = default;

                void forward(const Eigen::MatrixXf &in, Eigen::MatrixXf &out) override;

                void backward(const Eigen::MatrixXf &dout, Eigen::MatrixXf &ddout) override;
        };
    }; // namespace Activation
};     // namespace Tipousi