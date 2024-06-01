#include "layer/dense.hpp"

namespace Tipousi
{
    namespace Layer
    {
        Dense::Dense(int in_size, int out_size)
        {
            m_weights = Eigen::MatrixXf::Random(in_size, out_size);
            m_bias    = Eigen::MatrixXf::Random(1, out_size);
        }

        void Dense::forward(const Eigen::MatrixXf &in, Eigen::MatrixXf &out)
        {
            out = in * m_weights;
            // TODO: should the bias be substracted or added?
            // TODO: think about extendability, using 0 here is bad
            out.rowwise() += m_bias.row(0);
        }

        void Dense::backward(const Eigen::MatrixXf &dout,
                             Eigen::MatrixXf       &ddout)
        {
        }
    }  // namespace Layer
}  // namespace Tipousi