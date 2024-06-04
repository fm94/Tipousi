#include "data/dataset.hpp"

namespace Tipousi
{
    namespace Data
    {
        Dataset::Dataset(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y)
            : m_X(X), m_y(Y)
        {
        }

        Dataset::Iterator::Iterator(const Eigen::MatrixXd &X,
                                    const Eigen::MatrixXd &Y, size_t index)
            : m_X(X), m_y(Y), m_index(index)
        {
        }

        Dataset::Iterator &Dataset::Iterator::operator++()
        {
            ++m_index;
            return *this;
        }

        bool Dataset::Iterator::operator!=(const Dataset::Iterator &other) const
        {
            return m_index != other.m_index;
        }

        Dataset::DataPair Dataset::Iterator::operator*() const
        {
            Eigen::MatrixXf x = m_X.row(m_index).cast<float>();
            Eigen::MatrixXf y = m_y.row(m_index).cast<float>();
            return {x, y};
        }

        Dataset::Iterator Dataset::begin() const
        {
            return Iterator(m_X, m_y, 0);
        }

        Dataset::Iterator Dataset::end() const
        {
            return Iterator(m_X, m_y, m_X.rows());
        }
    }  // namespace Data
}  // namespace Tipousi