#pragma once

#include "base/op.hpp"

namespace Tipousi
{
    namespace Activation
    {
        class MyTemplate : public Op
        {
        public:
            MyTemplate();
            ~MyTemplate() override = default;

            void forward(const Eigen::MatrixXf &in, Eigen::MatrixXf &out) override;

            void backward(const Eigen::MatrixXf &dout, Eigen::MatrixXf &ddout) override;
        };
    }; // namespace Activation
}; // namespace Tipousi