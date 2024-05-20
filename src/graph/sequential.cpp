#include "graph/sequential.hpp"

namespace Tipousi
{
    namespace Graph
    {

        void Sequential::add_node(Node *node)
        {
            m_nodes.push_back(node);
        }

        void Sequential::forward()
        {
            //
        }

        void Sequential::backward()
        {
            //
        }

    }
}
