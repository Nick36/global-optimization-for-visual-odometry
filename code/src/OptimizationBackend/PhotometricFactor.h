#pragma once

#include <cmath>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include "FullSystem/FullSystem.h"

namespace gtsam
{
using namespace dso;

class PhotometricFactor: public NoiseModelFactor2<Pose3, Pose3>
{

private:

    typedef PhotometricFactor This;
    typedef NoiseModelFactor2<Pose3, Pose3> Base;

    FrameHessian* reference_;
    FrameHessian* target_;
    CalibHessian* Hcalib_;
    int lvl_;
    double scaleResidual_;

public:

    /** Constructor */
    PhotometricFactor (Key key1, Key key2, const SharedNoiseModel& model, FrameHessian* reference, FrameHessian* target, CalibHessian* Hcalib,
                       int lvl, double scaleResidual = 1.0) :
        Base(model, key1, key2), reference_(reference), target_(target), Hcalib_(Hcalib), lvl_(lvl), scaleResidual_(scaleResidual) {}

    virtual ~PhotometricFactor() {}

    /// @return a deep copy of this factor
    virtual gtsam::NonlinearFactor::shared_ptr clone() const
    {
        return boost::static_pointer_cast<gtsam::NonlinearFactor>(gtsam::NonlinearFactor::shared_ptr(new This(*this)));
    }

    /** implement functions needed for Testable */

    /** print */
    virtual void print(const std::string& s, const KeyFormatter& keyFormatter = DefaultKeyFormatter) const
    {
        std::cout << s << "PhotometricFactor("
                  << keyFormatter(this->key1()) << ","
                  << keyFormatter(this->key2()) << ")\n";
        this->noiseModel_->print("  noise model: ");
    }

    /** equals */
    virtual bool equals(const NonlinearFactor& expected, double tol=1e-9) const
    {
        const This *e =  dynamic_cast<const This*> (&expected);
        return e != NULL && Base::equals(*e, tol);
    }

    /** implement functions needed to derive from Factor */

    /** vector of errors */
    Vector evaluateError(const Pose3& p1, const Pose3& p2, boost::optional<Matrix&> H1 =
            boost::none, boost::optional<Matrix&> H2 = boost::none) const
    {
        std::vector<PointHessian*> points = reference_->pointHessiansMarginalized;
        std::vector<double> stdResidualVector;
        std::vector<double> stdGphi_aVector;

        Eigen::Vector3f* colorRef = reference_->dIp[lvl_];
        Eigen::Vector3f* colorNew = target_->dIp[lvl_];

        float scale = (float) (pow(0.5, lvl_));

        float fxl = Hcalib_->fxl() * scale;
        float fyl = Hcalib_->fyl() * scale;
        float cxl = (Hcalib_->cxl() + 0.5) / ((int)1<<lvl_) - 0.5;
        float cyl = (Hcalib_->cyl() + 0.5) / ((int)1<<lvl_) - 0.5;

        Mat33 K;
        K << fxl, 0.0, cxl, 0.0, fyl, cyl, 0.0, 0.0, 1.0;

        Eigen::Matrix<double,4,4> T = (p1.inverse() * p2).matrix();

        // generator matrices for SE3

        Eigen::Matrix<double,4,4> g1 = Eigen::Matrix<double,4,4>::Zero();
        g1 (1,2) = - 1.0;
        g1 (2,1) = 1.0;

        Eigen::Matrix<double,4,4> g2 = Eigen::Matrix<double,4,4>::Zero();
        g2 (2,0) = - 1.0;
        g2 (0,2) = 1.0;

        Eigen::Matrix<double,4,4> g3 = Eigen::Matrix<double,4,4>::Zero();
        g3 (0,1) = - 1.0;
        g3 (1,0) = 1.0;

        Eigen::Matrix<double,4,4> g4 = Eigen::Matrix<double,4,4>::Zero();
        g4 (0,3) = 1.0;

        Eigen::Matrix<double,4,4> g5 = Eigen::Matrix<double,4,4>::Zero();
        g5 (1,3) = 1.0;

        Eigen::Matrix<double,4,4> g6 = Eigen::Matrix<double,4,4>::Zero();
        g6 (2,3) = 1.0;

        Vec2f affL = reference_->targetPrecalc[target_->idx].PRE_aff_mode;

        unsigned int validResiduals = 0;

        for(int i=0;i<points.size();i++)
        {
            bool isGood = true;
            PointHessian* point = points[i];

            // all residuals in the pattern
            std::vector<double> patternResiduals;

            // all Jacobians with respect to T in the pattern
            std::vector<double> patternGphi_a;

            float u_scaled = point->u * scale;
            float v_scaled = point->v * scale;
            float idepth_scaled = point->idepth_scaled;

//            if (lvl_ == pyrLevelsUsed - 1) {
//            cout << "point coordinates and scale: " << u_scaled << " " << v_scaled << " " << scale << endl;
//            cin.get();
//            }

            // we have an offset of maximum two in the residual pattern
            // we also use the pixels one to the right, one to the bottom and one to both the right and the bottom
            if (!(u_scaled > 1 && v_scaled > 1 && u_scaled < wG[lvl_]-3 && v_scaled < hG[lvl_]-3))
            {
                isGood = false;
                //cout << "point out of frame" << endl;
            }
            else
                for (int idx=0;idx<patternNum;idx++)
                {
                    int dx = patternP[idx][0];
                    int dy = patternP[idx][1];

                    Vec3 pinit;
                    pinit << (u_scaled + dx) / idepth_scaled,
                             (v_scaled + dy) / idepth_scaled,
                             1.0 / idepth_scaled;

                    Vec4 p;
                    p << K.inverse() * pinit , 1.0;

                    Vec4 pt = T * p;
                    float u =  (pt[0] / pt[2]);
                    float v = (float) (pt[1] / pt[2]);
                    float Ku = fxl * u + cxl;
                    float Kv = fyl * v + cyl;
                    float new_idepth = (float) (1.0 / pt[2]);

                    // we have an offset of maximum two in the residual pattern
                    // we also use the pixels one to the right, one to the bottom and one to both the right and the bottom
                    if(!(Ku > 1 && Kv > 1 && Ku < wG[lvl_]-3 && Kv < hG[lvl_]-3 && new_idepth > 0))
                    {
                        isGood = false;
                        //cout << "projected point out of frame" << endl;
                        break;
                    }

                    Vec3f hitColor = getInterpolatedElement33(colorNew, Ku, Kv, wG[lvl_]);
                    float rlR = getInterpolatedElement31(colorRef, u_scaled + dx, v_scaled + dy, wG[lvl_]);

                    if(!std::isfinite(rlR) || !std::isfinite((float)hitColor[0]))
                    {
                        isGood = false;
                        break;
                    }

                    float residual = hitColor[0] - affL[0] * rlR - affL[1];

                    // Huber loss function
//                    float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
                    float hw = (fabs(residual) < setting_huberTH) ? 1 : 2 * (setting_huberTH / fabs(residual)) -
                                                                  (setting_huberTH / fabs(residual)) * (setting_huberTH / fabs(residual));

                    //patternResiduals.push_back(hw *residual*residual*(2-hw));
                    patternResiduals.push_back(sqrt(hw) * residual);

                    //if (hw < 1) hw = sqrtf(hw);

                    Eigen::Matrix<double,1,2> dI2Pi;
                    dI2Pi << hw*hitColor[1]*fxl, hw*hitColor[2]*fyl;

                    Eigen::Matrix<double,2,4> dPiP;
                    dPiP << 1/pt[2] , 0.0, -pt[0] / (pt[2] * pt[2]), 0.0, 0.0, 1/pt[2], -pt[1] / (pt[2] * pt[2]), 0.0;

                    // general pattern: dI2Pi * dPiP * T * gi * p
                    patternGphi_a.push_back(dI2Pi * dPiP * T * g1 * p);
                    patternGphi_a.push_back(dI2Pi * dPiP * T * g2 * p);
                    patternGphi_a.push_back(dI2Pi * dPiP * T * g3 * p);
                    patternGphi_a.push_back(dI2Pi * dPiP * T * g4 * p);
                    patternGphi_a.push_back(dI2Pi * dPiP * T * g5 * p);
                    patternGphi_a.push_back(dI2Pi * dPiP * T * g6 * p);
                }

            if(!isGood || std::accumulate(patternResiduals.begin(), patternResiduals.end(), 0.0) > 20 * patternNum * setting_outlierTH)
            {
                // insert max residual into residual vector
                double hw = 2 * (setting_huberTH / 256.0) -
                        (setting_huberTH / 256.0) * (setting_huberTH / 256.0);
                for (int j = 0; j < patternNum; j++) stdResidualVector.push_back(sqrt(hw) * 256.0);

                // insert zeros into T Jacobians vector
                for (int j = 0; j < 6 * patternNum; j++) stdGphi_aVector.push_back(0.0);

                continue;
            }

            for (double r : patternResiduals) assert (!(::isnan(r)));
            for (double gphi : patternGphi_a) assert (!(::isnan(gphi)));

            // insert pattern residuals into residual vector
            stdResidualVector.insert(stdResidualVector.end(), patternResiduals.begin(), patternResiduals.end());

            // insert T pattern Jacobians into T Jacobians vector
            stdGphi_aVector.insert(stdGphi_aVector.end(), patternGphi_a.begin(), patternGphi_a.end());

            validResiduals += patternNum;
        }

        //assert(validResiduals > 0);

        size_t numOfResisuals = stdResidualVector.size();

        cout << "keys " << key1() << " and " << key2() << ": " << numOfResisuals - validResiduals << "<>" << validResiduals << endl;

        // set H1 and H2
        Eigen::Matrix< double , Eigen::Dynamic, 6 > Gphi_a =
                Eigen::Map<Eigen::Matrix< double , Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> (stdGphi_aVector.data(), numOfResisuals, 6);
        assert (Gphi_a.allFinite());
        Matrix6 phi_prim_a = - (p2.inverse() * p1).AdjointMap();

        if (H1) (*H1) = ((scaleResidual_ * Gphi_a) / validResiduals) * phi_prim_a;
        if (H2) (*H2) = (scaleResidual_ * Gphi_a) / validResiduals;

        Eigen::Map<Vector> resVec (stdResidualVector.data(), numOfResisuals, 1);
        assert (resVec.allFinite());
        return (scaleResidual_ * resVec) / validResiduals;
    }

    /** return the measured */
    const Vector measured() const {
      return Vector::Zero (reference_->pointHessiansMarginalized.size());
    }

    /** number of variables attached to this factor */
    std::size_t size() const
    {
        return 2;
    }

}; // \class PhotometricFactor

} /// namespace gtsam
