#pragma once

#include "optimizer/base.hpp"

namespace Tipousi
{
    namespace Optimizer
    {
        class SGD : public OptimizerBase
        {
          public:
            SGD();
            ~SGD() = default;
        };
    };  // namespace Optimizer
};      // namespace Tipousi