#include "loss/mse.hpp"

#include "activation/relu.hpp"
#include "activation/softmax.hpp"

#include "layer/dense.hpp"

#include <memory>

#include "graph/node.hpp"
#include "graph/sequential.hpp"

using namespace Tipousi;
using namespace Graph;
using namespace Layer;
using namespace Activation;
using namespace Loss;

void test_create_net()
{
    // create layer nodes
    // these are raw ptrs and ownership will go to the graph,
    // it is responsable for cleaning them!
    Node *node1 = Node::create<Dense>(5, 32);
    Node *node2 = Node::create<ReLU>();
    Node *node3 = Node::create<Dense>(32, 1);
    Node *node4 = Node::create<Softmax>();

    // build the dependencies
    node2->add_input(node1); // node2 depends on node1
    node3->add_input(node2); // node3 depends on node2
    node4->add_input(node3); // node4 depends on node2

    // create the graph
    Sequential net;
    net.add_node(node1);
    net.add_node(node2);
    net.add_node(node3);
    net.add_node(node4);

    // forward and backward pass
    net.forward();
    net.backward();
}

int main(int argc, char const *argv[])
{
    // tests

    // loss functions
    MSE mse;

    // activation functions
    ReLU relu;
    Softmax softmax;

    // layers
    Dense dense(16, 32); // 16 inputs, 32 outputs

    // build a model - current idea
    test_create_net();
}
