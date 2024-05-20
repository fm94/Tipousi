#include <vector>

#include "model/node.hpp"

namespace Tipousi
{
    namespace Graph
    {
        class Sequential
        {

        public:
            Sequential() = default;
            ~Sequential()
            {
                for (auto node : m_nodes)
                {
                    delete node;
                }
            }

            void add_node(Graph::Node *node);
            std::vector<float> forward();
            void backward();

        private:
            std::vector<Graph::Node *> m_nodes;
        };

    }
}
