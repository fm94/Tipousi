#pragma once

#include "base/op.hpp"
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
            void backward(const Eigen::MatrixXf &dout, Eigen::MatrixXf &ddout);

            void add_input(Node *node);
            void add_output(Node *node);

            std::vector<Node *> &get_outputs() { return m_outputs; }
            std::vector<Node *> &get_inputs() { return m_inputs; }

            void set_learning_rate(float learning_rate)
            {
                m_operation->set_learning_rate(learning_rate);
            }

          private:
            Node(std::unique_ptr<Op> ptr);

            std::unique_ptr<Op> m_operation;
            std::vector<Node *> m_inputs;
            std::vector<Node *> m_outputs;
        };
    }  // namespace Graph
}  // namespace Tipousi