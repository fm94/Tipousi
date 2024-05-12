#pragma once

#include "base/op.hpp"

#include <iostream>
#include <vector>

namespace Tipousi
{
    namespace Model
    {
        class Sequential
        {
        public:
            Sequential(std::vector<Op *> &model);
            ~Sequential() = default;

            void forward(Eigen::MatrixXf &x);

            void backward(float &loss, const Eigen::MatrixXf &true_y, Eigen::MatrixXf &pred_y);

        private:
            std::vector<Op *> m_model;
        };
    }; // namespace Model
}; // namespace Tipousi