#pragma once

inline auto alpha() {
    return this->block(0, 0, this->rows() / 2, this->cols());
}
inline auto beta() {
    return this->block(this->rows() / 2, 0, this->rows() / 2, this->cols());
}
inline auto alpha_alpha() {
    return this->block(0, 0, this->rows() / 2, this->cols() / 2);
}
inline auto alpha_beta() {
    return this->block(this->rows() / 2, 0, this->rows() / 2, this->cols() / 2);
}
inline auto beta_alpha() {
    return this->block(0, this->cols() / 2, this->rows() / 2, this->cols() / 2);
}
inline auto beta_beta() {
    return this->block(this->rows() / 2, this->cols() / 2, this->rows() / 2,
                       this->cols() / 2);
}
inline const auto alpha() const {
    return this->block(0, 0, this->rows() / 2, this->cols());
}
inline const auto beta() const {
    return this->block(this->rows() / 2, 0, this->rows() / 2, this->cols());
}
inline const auto alpha_alpha() const {
    return this->block(0, 0, this->rows() / 2, this->cols() / 2);
}
inline const auto alpha_beta() const {
    return this->block(this->rows() / 2, 0, this->rows() / 2, this->cols() / 2);
}
inline const auto beta_alpha() const {
    return this->block(0, this->cols() / 2, this->rows() / 2, this->cols() / 2);
}
inline const auto beta_beta() const {
    return this->block(this->rows() / 2, this->cols() / 2, this->rows() / 2,
                       this->cols() / 2);
}
