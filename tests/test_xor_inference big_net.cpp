#include "activation/relu.hpp"
#include "activation/sigmoid.hpp"
#include "data/dataset.hpp"
#include "graph/node.hpp"
#include "graph/sequential.hpp"
#include "layer/dense.hpp"
#include "loss/mse.hpp"
#include "optimizer/adam.hpp"
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

TEST(BigNetTest, DISABLED_XORTest)
{
    // in this test we try to train a net to learn the xor operation
    // we have two inputs and one output
    int n_features{2};
    int n_labels{1};

    Node *node1 = Node::create<Dense>(n_features, 30);
    Node *node2 = Node::create<Sigmoid>();
    Node *node3 = Node::create<Dense>(30, 30);
    Node *node4 = Node::create<Sigmoid>();
    Node *node5 = Node::create<Dense>(30, n_labels);
    Node *node6 = Node::create<Sigmoid>();

    // build the dependencies
    node2->add_input(node1);
    node3->add_input(node2);
    node4->add_input(node3);
    node5->add_input(node4);
    node6->add_input(node5);

    float learning_rate{0.1f};
    Adam  optimizer(learning_rate);

    // create the graph (pass input and output nodes)
    Sequential net(node1, node6, optimizer);
    net.summary();

    // test inference
    Eigen::MatrixXf X(4, 2);
    Eigen::MatrixXf Y(4, 1);
    // XOR inputs
    X << 0, 0, 0, 1, 1, 0, 1, 1;
    // XOR outputs (labels)
    Y << 0, 1, 1, 0;

    // create dataset
    size_t  batch_size = 4;
    Dataset dataset(X, Y, batch_size);

    // define the loss
    MSE  mse;
    auto start = std::chrono::high_resolution_clock::now();
    net.train(dataset, mse, 10000);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "Training time: " << duration << " microseconds" << std::endl;

    // forward pass with time measurement
    Eigen::MatrixXf preds;
    start = std::chrono::high_resolution_clock::now();
    net.forward(X, preds);
    end = std::chrono::high_resolution_clock::now();
    duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "Forward pass execution time: " << duration << " microseconds"
              << std::endl;

    // compute the loss
    auto loss = mse.compute(Y, preds);
    std::cout << "The loss of the network after training: " << loss
              << std::endl;

    std::cout << "gt: " << Y << std::endl;
    std::cout << "predictions: " << preds << std::endl;
    EXPECT_LT(loss, 0.5f);  // better than random guessing
}