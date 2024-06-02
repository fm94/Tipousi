#include "activation/relu.hpp"
#include "activation/softmax.hpp"
#include "graph/node.hpp"
#include "graph/sequential.hpp"
#include "layer/dense.hpp"
#include <chrono>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>

using namespace Tipousi;
using namespace Graph;
using namespace Layer;
using namespace Activation;

TEST(SimpleNetTest, SimpleCreation)
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
    EXPECT_NO_THROW(net.forward(features, preds));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();

    std::cout << "Forward pass execution time: " << duration << " microseconds"
              << std::endl;

    // backward pass
    EXPECT_NO_THROW(net.backward());
}