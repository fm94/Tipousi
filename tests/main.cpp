#include "loss/mse.hpp"

#include "activation/relu.hpp"
#include "activation/softmax.hpp"

int main(int argc, char const *argv[])
{
    // tests

    // loss functions
    Tipousi::Loss::MSE mse;

    // activation functions
    Tipousi::Activation::ReLU relu;
    Tipousi::Activation::Softmax softmax;
}
