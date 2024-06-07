#include "activation/relu.hpp"
#include "activation/softmax.hpp"
#include "data/dataset.hpp"
#include "graph/node.hpp"
#include "graph/sequential.hpp"
#include "layer/dense.hpp"
#include "loss/mse.hpp"
#include "optimizer/sgd.hpp"
#include <chrono>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>

using namespace Tipousi;
using namespace Graph;
using namespace Layer;
using namespace Activation;
using namespace Loss;
using namespace Data;
using namespace Optimizer;

TEST(SimpleNetTest, XORTest)
{
    // in this test we try to train a net to lear the xor operation
    // we have two inputs and one output
    int n_features{2};
    int n_labels{1};

    Node *node1 = Node::create<Dense>(n_features, 32);
    Node *node2 = Node::create<ReLU>();
    Node *node3 = Node::create<Dense>(32, n_labels);
    Node *node4 = Node::create<Softmax>();

    // build the dependencies
    node2->add_input(node1);  // node2 depends on node1
    node3->add_input(node2);  // node3 depends on node2
    node4->add_input(node3);  // node4 depends on node2

    // create the graph (pass input and output nodes)
    float      learning_rate{0.001f};
    Sequential net(node1, node4, learning_rate);

    // test inference
    Eigen::MatrixXf X(4, 2);
    Eigen::MatrixXf Y(4, 1);
    // XOR inputs
    X << 0, 0, 0, 1, 1, 0, 1, 1;
    // XOR outputs (labels)
    Y << 0, 1, 1, 0;

    Eigen::MatrixXf preds;
    net.forward(X, preds);

    // create dataset
    // TODO add eppochs etc
    Dataset dataset(X, Y);

    // define the optimizer and the loss
    SGD sgd;
    MSE mse;
    net.train(dataset, sgd, mse, 10);

    // forward pass with time measurement
    // Eigen::MatrixXf preds;
    auto            start = std::chrono::high_resolution_clock::now();
    net.forward(X, preds);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "Forward pass execution time: " << duration << " microseconds"
              << std::endl;

    // compute the loss
    std::cout << "The loss of the network after training: " << mse.compute(Y, preds) << std::endl;
}