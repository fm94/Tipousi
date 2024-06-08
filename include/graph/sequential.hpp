#pragma once

#include "graph/node.hpp"
#include "graph/trainable.hpp"
#include "optimizer/base.hpp"
#include <vector>

namespace Tipousi
{
    namespace Graph
    {
        class Sequential : public Trainable
        {

          public:
            Sequential(Node *input_node, Node *output_node,
                       Optimizer::OptimizerBase &optimizer);

            ~Sequential()
            {
                // travers nodes in the registry
                // and delete all of them sequentially
                for (Node *node : m_node_registry)
                {
                    delete node;
                }
            }

            void forward(const Eigen::MatrixXf &in, Eigen::MatrixXf &out);
            void backward(Eigen::MatrixXf &initial_grads);

            void train(const Data::Dataset &dataset, const Loss::LossBase &loss,
                       const uint32_t n_epochs) override;

          private:
            Node               *m_input_node  = nullptr;
            Node               *m_output_node = nullptr;
            std::vector<Node *> m_node_registry{};
        };

    }  // namespace Graph
}  // namespace Tipousi
