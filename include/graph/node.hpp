#pragma once

#include <vector>
#include <memory>

#include "base/op.hpp"

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
                return new Node(std::make_unique<T>(std::forward<Args>(args)...));
            }

            void forward(const Eigen::MatrixXf &in, Eigen::MatrixXf &out);
            void backward(const Eigen::MatrixXf &dout, Eigen::MatrixXf &ddout);

            void add_input(Node *node);

        private:
            Node(std::unique_ptr<Op> ptr);

            std::unique_ptr<Op> m_operation;
            std::vector<Node *> m_inputs;
            std::vector<Node *> m_outputs;
        };
    }
}