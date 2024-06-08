#include "graph/sequential.hpp"
#include <iostream>

namespace Tipousi
{
    namespace Graph
    {
        Sequential::Sequential(Node *input_node, Node *output_node,
                               Optimizer::OptimizerBase *optimizer)
            : m_input_node(input_node), m_output_node(output_node)
        {
            // mechanism to register all nodes
            // to handle destruction correction at the end
            Node *current_node = m_input_node;
            while (true)
            {
                if (current_node)
                {
                    current_node->set_optimizer(optimizer);
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
            // in should be const?
            //
            // copy to create the object that will be passed through the network
            // while keeping the original data intact
            Eigen::MatrixXf data_copy    = in;
            Node           *current_node = m_input_node;
            // continue until no more nodes
            while (true)
            {
                if (current_node)
                {
                    current_node->forward(data_copy);
                    // TODO hacky approachs: always take number 0
                    auto &output_nodes = current_node->get_outputs();
                    if (output_nodes.size() == 0 || !output_nodes[0])
                    {
                        break;
                    }
                    current_node = output_nodes[0];
                }
            }
            out = data_copy;  // copying happening?
        }

        void Sequential::backward(Eigen::MatrixXf &initial_grads)
        {
            // think if copying is really needed here
            Eigen::MatrixXf grad_copy    = initial_grads;
            Node           *current_node = m_output_node;
            // continue until no more nodes
            while (true)
            {
                if (current_node)
                {
                    current_node->backward(grad_copy);
                    // TODO hacky approachs: always take number 0
                    auto &input_nodes = current_node->get_inputs();
                    if (input_nodes.size() == 0 || !input_nodes[0])
                    {
                        break;
                    }
                    current_node = input_nodes[0];
                }
            }
        }

        void Sequential::train(const Data::Dataset  &dataset,
                               const Loss::LossBase &loss_func,
                               const uint32_t        n_epochs)
        {
            for (uint32_t i{0}; i < n_epochs; i++)
            {
                float    total_loss = 0.0f;
                uint32_t counter{0};
                for (const auto &[x, y] : dataset)
                {
                    Eigen::MatrixXf output;
                    Eigen::MatrixXf out_grad;
                    forward(x, output);
                    total_loss += loss_func.compute(y, output);
                    loss_func.grad(out_grad, y, output);
                    // std::cout << "gt " << y << std::endl;
                    // std::cout << "pred " << output << std::endl;
                    // std::cout << "loss " << total_loss << std::endl;
                    // std::cout << "grad " << out_grad << std::endl;
                    backward(out_grad);
                    counter++;
                }
                std::cout << "Epoch: " << i
                          << ", Loss: " << total_loss / counter << std::endl;
            }
        }

    }  // namespace Graph
}  // namespace Tipousi
