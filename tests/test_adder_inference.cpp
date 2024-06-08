#include "activation/relu.hpp"
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

TEST(SimpleNetTest, AdderTest)
{
    // in this test we try to train a net to learn the addition operation
    // we have two inputs and one output
    int n_features{2};
    int n_labels{1};

    Node *node1 = Node::create<Dense>(n_features, 16);
    Node *node2 = Node::create<ReLU>();
    Node *node3 = Node::create<Dense>(16, n_labels);

    // build the dependencies
    node2->add_input(node1);
    node3->add_input(node2);

    float learning_rate{0.01f};
    SGD   sgd(learning_rate);

    // create the graph (pass input and output nodes)
    Sequential net(node1, node3, &sgd);

    // test inference
    Eigen::MatrixXf X(4, 2);
    Eigen::MatrixXf Y(4, 1);

    // y is just the summation of two x inputs
    X << 1, 1, 1, 2, 2, 1, 2, 2;
    Y << 2, 3, 3, 4;

    // create dataset
    Dataset dataset(X, Y);

    // define the loss
    MSE  mse;
    auto start = std::chrono::high_resolution_clock::now();
    net.train(dataset, mse, 50);
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
    EXPECT_LT(loss, 0.02f);
}