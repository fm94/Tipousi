#pragma once

#include <vector>

#include "graph/node.hpp"

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

            void add_node(Node *node);
            void forward();
            void backward();

        private:
            std::vector<Node *> m_nodes;
        };

    }
}
