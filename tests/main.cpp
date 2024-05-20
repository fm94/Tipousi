#include "loss/mse.hpp"

#include "activation/relu.hpp"
#include "activation/softmax.hpp"

#include "layer/dense.hpp"

#include <memory>

#include "model/node.hpp"
#include "model/sequential.hpp"

using namespace Tipousi;


void test_create_net(){
    // create layer nodes
    // these are raw ptrs and ownership will go to the graph,
    // it is responsable for cleaning them!
    Graph::Node* node1 = Graph::Node::create<Layer::Dense>(5, 32);
    Graph::Node* node2 = Graph::Node::create<Activation::ReLU>();
    Graph::Node* node3 = Graph::Node::create<Layer::Dense>(32, 1);
    Graph::Node* node4 = Graph::Node::create<Activation::Softmax>();

    // build the dependencies
    node2->add_input(node1); // node2 depends on node1
    node3->add_input(node2); // node3 depends on node2
    node4->add_input(node3); // node4 depends on node2

    // create the graph
    Graph::Sequential net;
    net.add_node(node1);
    net.add_node(node2);
    net.add_node(node3);
    net.add_node(node4);

    // forward and backward pass
    std::vector<float> output = net.forward();
    net.backward();
}

int main(int argc, char const *argv[])
{
    // tests

    // loss functions
    Loss::MSE mse;

    // activation functions
    Activation::ReLU relu;
    Activation::Softmax softmax;

    // layers
    Layer::Dense dense(16, 32); // 16 inputs, 32 outputs

    // build a model - current idea
    test_create_net();
}
