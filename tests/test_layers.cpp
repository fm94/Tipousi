#include "activation/relu.hpp"
#include "activation/softmax.hpp"
#include "layer/dense.hpp"
#include "loss/mse.hpp"
#include <gtest/gtest.h>

using namespace Tipousi;
using namespace Layer;
using namespace Activation;
using namespace Loss;

TEST(OpTests, SimpleCreation)
{
    // loss functions
    MSE mse;

    // activation functions
    ReLU    relu;
    Softmax softmax;

    // layers
    Dense dense(16, 32);  // 16 inputs, 32 outputs
}