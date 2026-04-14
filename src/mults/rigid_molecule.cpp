#include <occ/mults/rigid_molecule.h>
#include <Eigen/Geometry>
#include <cmath>

namespace occ::mults {

Mat3 RigidMolecule::proper_rotation_matrix() const {
    double angle = angle_axis.norm();
    if (angle < 1e-12) {
        return Mat3::Identity();
    }
    Vec3 axis = angle_axis / angle;
    return Eigen::AngleAxisd(angle, axis).toRotationMatrix();
}

Mat3 RigidMolecule::rotation_matrix() const {
    return static_cast<double>((parity < 0) ? -1 : 1) *
           proper_rotation_matrix();
}

void RigidMolecule::set_from_rotation(RigidMolecule &mol, const Vec3 &pos,
                                       const Mat3 &R) {
    mol.com = pos;

    // Determine parity from determinant
    double det = R.determinant();
    mol.parity = (det < 0) ? -1 : 1;

    // Extract proper rotation
    Mat3 Q = static_cast<double>(mol.parity) * R;

    // Re-orthogonalize if needed
    const Mat3 I = Mat3::Identity();
    if ((Q.transpose() * Q - I).norm() > 1e-6) {
        Eigen::JacobiSVD<Mat3> svd(Q, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Q = svd.matrixU() * svd.matrixV().transpose();
    }

    // Extract angle-axis
    Eigen::AngleAxisd aa(Q);
    double angle = aa.angle();
    if (angle < 1e-12) {
        mol.angle_axis = Vec3::Zero();
    } else {
        mol.angle_axis = aa.axis() * angle;
    }
}

} // namespace occ::mults
