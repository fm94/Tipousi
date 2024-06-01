#include "activation/relu.hpp"
#include "activation/softmax.hpp"
#include "graph/node.hpp"
#include "graph/sequential.hpp"
#include "layer/dense.hpp"
#include "loss/mse.hpp"
#include <chrono>
#include <iostream>
#include <memory>

using namespace Tipousi;
using namespace Graph;
using namespace Layer;
using namespace Activation;
using namespace Loss;

void test_create_net()
{
    int n_features{2};
    int n_labels{2};

    // create layer nodes
    // these are raw ptrs and ownership will go to the graph,
    // it is responsable for cleaning them!
    Node *node1 = Node::create<Dense>(n_features, 32);
    Node *node2 = Node::create<ReLU>();
    Node *node3 = Node::create<Dense>(32, n_labels);
    Node *node4 = Node::create<Softmax>();

    // build the dependencies
    node2->add_input(node1);  // node2 depends on node1
    node3->add_input(node2);  // node3 depends on node2
    node4->add_input(node3);  // node4 depends on node2

    // create the graph (pass input and output nodes)
    Sequential net(node1, node4);

    // test inference
    int  n_samples{32};
    auto features = Eigen::MatrixXf::Random(n_samples, n_features);
    auto labels   = Eigen::MatrixXf(n_samples, n_labels);

    // forward pass with time measurement
    Eigen::MatrixXf preds;
    auto            start = std::chrono::high_resolution_clock::now();
    net.forward(features, preds);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();

    std::cout << "Forward pass execution time: " << duration << " microseconds"
              << std::endl;

    // backward pass
    net.backward();
}

int main(int argc, char const *argv[])
{
    // tests

    // loss functions
    MSE mse;

    // activation functions
    ReLU    relu;
    Softmax softmax;

    // layers
    Dense dense(16, 32);  // 16 inputs, 32 outputs

    // build a model - current idea
    test_create_net();
}
