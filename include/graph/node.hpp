#pragma once

#include "base/op.hpp"
#include "optimizer/base.hpp"
#include <memory>
#include <vector>

namespace Tipousi
{
    namespace Graph
    {
        class Node
        {
          public:
            template <typename T, typename... Args>
            static Node *create(Args &&...args)
            {
                return new Node(
                    std::make_unique<T>(std::forward<Args>(args)...));
            }

            void forward(Eigen::MatrixXf &data);
            void backward(Eigen::MatrixXf &grads);

            void add_input(Node *node);
            void add_output(Node *node);

            void set_optimizer(Optimizer::OptimizerBase *optimizer);

            std::vector<Node *> &get_outputs() { return m_outputs; }
            std::vector<Node *> &get_inputs() { return m_inputs; }

          private:
            Node(std::unique_ptr<Op> ptr);

            std::unique_ptr<Op> m_operation;
            std::vector<Node *> m_inputs;
            std::vector<Node *> m_outputs;
        };
    }  // namespace Graph
}  // namespace Tipousi