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
            template<typename T, typename... Args>
            static Node* create(Args&&... args);

            std::vector<float> forward(std::vector<float> input_data);
            void backward(std::vector<float> grad_output);

            void add_input(Node* node);

        private:
            Node(std::unique_ptr<Op> ptr);

            std::unique_ptr<Op> m_operation;
            std::vector<Node *> m_inputs;
            std::vector<Node *> m_outputs;
        };
    }
}