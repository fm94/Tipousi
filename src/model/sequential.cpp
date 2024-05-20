#include <vector>

#include "model/sequential.hpp"

namespace Tipousi
{
    namespace Graph
    {

        void Sequential::add_node(Graph::Node *node)
        {
            m_nodes.push_back(node);
        }

        std::vector<float> Sequential::forward()
        {
            //
        }

        void Sequential::backward()
        {
            //
        }

    }
}
