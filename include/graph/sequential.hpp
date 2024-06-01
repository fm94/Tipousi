#pragma once

#include "graph/node.hpp"
#include <vector>

namespace Tipousi
{
    namespace Graph
    {
        class Sequential
        {

          public:
            Sequential(Node *input_node, Node *output_node);

            ~Sequential()
            {
                // travers all nodes in backward pass
                // and delete all of them sequentially
                Node *current_node = m_output_node;
                while (true)
                {
                    // TODO : hacky approch deleting only first node
                    // -> should be recursive
                    auto input_nodes = current_node->get_inputs();
                    if (input_nodes.size() == 0 || !input_nodes[0])
                    {
                        break;
                    }
                    delete current_node;
                    current_node = input_nodes[0];
                }
            }

            void forward(const Eigen::MatrixXf &in, Eigen::MatrixXf &out);
            void backward();

          private:
            Node *m_input_node  = nullptr;
            Node *m_output_node = nullptr;
        };

    }  // namespace Graph
}  // namespace Tipousi
