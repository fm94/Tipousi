
#include "model/node.hpp"
#include "node.hpp"

namespace Tipousi
{
    namespace Graph
    {

        Node::Node(std::unique_ptr<Op> ptr) : m_operation(std::move(ptr)) {}

        template<typename T, typename... Args>
        Node* Node::create(Args&&... args) {
            return new Node(std::make_unique<T>(std::forward<Args>(args)...));
        }

        std::vector<float> Node::forward(std::vector<float> input_data)
        {
            for (auto *input_node : m_inputs)
            {
                auto input_result = input_node->forward({});
                input_data.insert(input_data.end(), input_result.begin(), input_result.end());
            }
            return m_operation->forward(input_data);
        }

        void Node::backward(std::vector<float> grad_output)
        {
            std::vector<float> grad_input = m_operation->backward(grad_output);
            for (auto *input_node : m_inputs)
            {
                input_node->backward(grad_input);
            }
        }

        void Node::add_input(Node *node)
        {
            m_inputs.push_back(node);
        }

    }
}

