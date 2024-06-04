#pragma once

#include "data/dataset.hpp"
#include "loss/base.hpp"
#include "optimizer/base.hpp"

namespace Tipousi
{
    namespace Graph
    {
        class Trainable
        {
          public:
            virtual void train(const Data::Dataset            &dataset,
                               const Optimizer::OptimizerBase &optimizer,
                               const Loss::LossBase           &loss,
                               const uint32_t                  n_epochs) = 0;

          protected:
            Trainable()          = default;
            virtual ~Trainable() = default;
        };

    }  // namespace Graph
}  // namespace Tipousi