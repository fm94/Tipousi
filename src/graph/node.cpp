#include "graph/node.hpp"

namespace Tipousi
{
    namespace Graph
    {
        Node::Node(std::unique_ptr<Op> ptr) : m_operation(std::move(ptr)) {}

        void Node::forward(Eigen::MatrixXf &data)
        {
            // for (auto *input_node : m_inputs)
            // {
            //     auto input_result = input_node->forward({});
            //     input_data.insert(input_data.end(), input_result.begin(),
            //     input_result.end());
            // }

            // TODO this is a fake call that is used only if the node has one Op
            m_operation->forward(data, data);

            // if we have mutliple inputs then save forward computations by
            // caching the current data. Experimental!!!
            if (m_outputs.size() > 1)
            {
                // m_cache
            }
        }

        void Node::backward(const Eigen::MatrixXf &dout, Eigen::MatrixXf &ddout)
        {
            // std::vector<float> grad_input =
            // m_operation->backward(grad_output); for (auto *input_node :
            // m_inputs)
            // {
            //     input_node->backward(grad_input);
            // }

            // TODO this is a fake call that is used only if the node has one Op
            m_operation->backward(dout, ddout);
        }

        void Node::add_input(Node *node)
        {
            node->add_output(this);
            m_inputs.push_back(node);
        }

        void Node::add_output(Node *node) { m_outputs.push_back(node); }

    }  // namespace Graph
}  // namespace Tipousi
