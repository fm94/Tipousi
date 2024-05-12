#include "model/sequential.hpp"

namespace Tipousi{
    namespace Model{
Sequential::Sequential(std::vector<Op *> &model): m_model(model) {
}

void Sequential::forward(Eigen::MatrixXf &x) {
}

void Sequential::backward(float &loss, const Eigen::MatrixXf &true_y, Eigen::MatrixXf &pred_y) {
}


    };
};



