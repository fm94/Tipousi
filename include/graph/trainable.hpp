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
            virtual void train(Data::Dataset        &dataset,
                               const Loss::LossBase &loss,
                               const uint32_t        n_epochs) = 0;

          protected:
            Trainable()          = default;
            virtual ~Trainable() = default;
        };

    }  // namespace Graph
}  // namespace Tipousi