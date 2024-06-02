#include "graph/sequential.hpp"

namespace Tipousi
{
    namespace Graph
    {
        Sequential::Sequential(Node *input_node, Node *output_node)
            : m_input_node(input_node), m_output_node(output_node)
        {
            // mechanism to register all nodes
            // to handle destruction correction at the end
            Node *current_node = m_input_node;
            while (true)
            {
                if (current_node)
                {
                    m_node_registry.push_back(current_node);
                    // TODO hacky approachs: always take number 0
                    auto &output_nodes = current_node->get_outputs();
                    if (output_nodes.size() == 0 || !output_nodes[0])
                    {
                        break;
                    }
                    current_node = output_nodes[0];
                }
            }
        }

        void Sequential::forward(const Eigen::MatrixXf &in,
                                 Eigen::MatrixXf       &out)
        {
            // copy to create the object that will be passed through the network
            // while keeping the original data intact
            Eigen::MatrixXf data_copy    = in;
            Node           *current_node = m_input_node;
            // continue until no more nodes
            while (true)
            {
                if (current_node)
                {
                    // TODO hacky approachs: always take number 0
                    auto &output_nodes = current_node->get_outputs();
                    if (output_nodes.size() == 0 || !output_nodes[0])
                    {
                        break;
                    }
                    current_node->forward(data_copy);
                    current_node = output_nodes[0];
                }
            }
            out = data_copy;  // copying happening?
        }

        void Sequential::backward()
        {
            //
        }

    }  // namespace Graph
}  // namespace Tipousi
