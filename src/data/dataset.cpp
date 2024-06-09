#include "data/dataset.hpp"

namespace Tipousi
{
    namespace Data
    {
        Dataset::Dataset(const Eigen::MatrixXf &X, const Eigen::MatrixXf &Y,
                         size_t batch_size)
            : m_X(X), m_y(Y), m_batch_size(batch_size)
        {
            if (batch_size > static_cast<size_t>(X.rows()))
            {
                throw std::invalid_argument(
                    "Batch size cannot be larger than the number of samples in "
                    "the dataset");
            }
        }

        Dataset::Iterator::Iterator(const Eigen::MatrixXf &X,
                                    const Eigen::MatrixXf &Y, size_t index,
                                    size_t batch_size)
            : m_X(X), m_y(Y), m_index(index), m_batch_size(batch_size)
        {
        }

        Dataset::Iterator &Dataset::Iterator::operator++()
        {
            m_index += m_batch_size;
            return *this;
        }

        bool Dataset::Iterator::operator!=(const Dataset::Iterator &other) const
        {
            // return m_index != other.m_index;
            //  drop last incomplete batch
            return m_index < other.m_index;
        }

        Dataset::DataPair Dataset::Iterator::operator*() const
        {
            // doing min for each batch? check this
            size_t          endIndex = std::min(m_index + m_batch_size,
                                                static_cast<size_t>(m_X.rows()));
            Eigen::MatrixXf x =
                m_X.block(m_index, 0, endIndex - m_index, m_X.cols());
            Eigen::MatrixXf y =
                m_y.block(m_index, 0, endIndex - m_index, m_y.cols());
            return {x, y};
        }

        Dataset::Iterator Dataset::begin() const
        {
            return Iterator(m_X, m_y, 0, m_batch_size);
        }

        Dataset::Iterator Dataset::end() const
        {
            // return Iterator(m_X, m_y, m_X.rows(), m_batch_size);
            // drop last incomplete batch
            size_t endIndex =
                (m_X.rows() / m_batch_size) *
                m_batch_size;  // Determine the last complete batch
            return Iterator(m_X, m_y, endIndex, m_batch_size);
        }

        void Dataset::shuffle()
        {
            std::vector<int> indices(m_X.rows());
            std::iota(indices.begin(), indices.end(), 0);
            std::random_device rd;
            std::mt19937       g(rd());
            std::shuffle(indices.begin(), indices.end(), g);

            Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(
                indices.size());
            for (size_t i = 0; i < indices.size(); ++i)
            {
                perm.indices()[i] = indices[i];
            }

            m_X = perm * m_X;
            m_y = perm * m_y;
        }
    }  // namespace Data
}  // namespace Tipousi