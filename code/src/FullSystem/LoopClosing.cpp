#include "FullSystem/FullSystem.h"
 
#include "stdio.h"
#include "util/globalFuncs.h"

#include "IOWrapper/Output3DWrapper.h"

#include "FullSystem/CoarseInitializer.h"

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>

// GTSAM
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Key.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>

#include "OptimizationBackend/PhotometricFactor.h"

#include <chrono>

namespace dso
{

void FullSystem::reorderMatches(
        const vector<DBoW2::FORB::TDescriptor> &A, const vector<unsigned int> &i_A,
        const vector<DBoW2::FORB::TDescriptor> &B, const vector<unsigned int> &i_B,
        vector<unsigned int> &i_match_A, vector<unsigned int> &i_match_B
        )
{
    double max_neighbour_ratio = 0.6;

    i_match_A.resize(0);
    i_match_B.resize(0);
    i_match_A.reserve( min(i_A.size(), i_B.size()) );
    i_match_B.reserve( min(i_A.size(), i_B.size()) );

    vector<unsigned int>::const_iterator ait, bit;
    unsigned int i, j;
    i = 0;
    for(ait = i_A.begin(); ait != i_A.end(); ++ait, ++i)
    {
        int best_j_now = -1;
        double best_dist_1 = 1e9;
        double best_dist_2 = 1e9;

        j = 0;
        for(bit = i_B.begin(); bit != i_B.end(); ++bit, ++j)
        {
            double d = DBoW2::FORB::distance(A[*ait], B[*bit]);

            // in i
            if(d < best_dist_1)
            {
                best_j_now = j;
                best_dist_2 = best_dist_1;
                best_dist_1 = d;
            }
            else if(d < best_dist_2)
            {
                best_dist_2 = d;
            }
        }

        if(best_dist_1 / best_dist_2 <= max_neighbour_ratio)
        {
            unsigned int idx_B = i_B[best_j_now];
            bit = find(i_match_B.begin(), i_match_B.end(), idx_B);

            if(bit == i_match_B.end())
            {
                i_match_B.push_back(idx_B);
                i_match_A.push_back(*ait);
            }
            else
            {
                unsigned int idx_A = i_match_A[ bit - i_match_B.begin() ];
                double d = DBoW2::FORB::distance(A[idx_A], B[idx_B]);
                if(best_dist_1 < d)
                {
                    i_match_A[ bit - i_match_B.begin() ] = *ait;
                }
            }

        }
    }
}

cv::Mat FullSystem::computeFundamentalMatrix(
        DBoW2::EntryId old_entry, const std::vector<cv::KeyPoint> &keys,
        const ORBFeatures &frameFeatures,
        const DBoW2::FeatureVector &fv,
        cv::Mat &framePoints, cv::Mat &matchPoints)
{
    int min_Fpoints = 8; //12
    int max_ransac_iterations = 500;
    double ransac_probability = 0.99;
    double max_reprojection_error = 50.0; // adjust if no fundamental matrix gets computed

    DVision::FSolver fsolver;
    fsolver.setImageSize(wG[0], hG[0]);

    const DBoW2::FeatureVector &oldvec = db->retrieveFeatures(old_entry);

    // for each word in common, get the closest descriptors

    vector<unsigned int> i_old, i_cur;

    DBoW2::FeatureVector::const_iterator old_it, cur_it;
    const DBoW2::FeatureVector::const_iterator old_end = oldvec.end();
    const DBoW2::FeatureVector::const_iterator cur_end = fv.end();

    old_it = oldvec.begin();
    cur_it = fv.begin();

    while(old_it != old_end && cur_it != cur_end)
    {
        if(old_it->first == cur_it->first)
        {
            // compute matches between
            // features old_it->second of image_keys[old_entry] and
            // features cur_it->second of keys
            vector<unsigned int> i_old_now, i_cur_now;

            reorderMatches(keyframeFeatures[old_entry], old_it->second, frameFeatures, cur_it->second, i_old_now, i_cur_now);

            i_old.insert(i_old.end(), i_old_now.begin(), i_old_now.end());
            i_cur.insert(i_cur.end(), i_cur_now.begin(), i_cur_now.end());

            // move old_it and cur_it forward
            ++old_it;
            ++cur_it;
        }
        else if(old_it->first < cur_it->first)
        {
            // move old_it forward
            old_it = oldvec.lower_bound(cur_it->first);
        }
        else
        {
            // move cur_it forward
            cur_it = fv.lower_bound(old_it->first);
        }
    }

    // calculate now the fundamental matrix
    if((int)i_old.size() >= min_Fpoints)
    {
        vector<cv::Point2f> old_points, cur_points;

        // add matches to the vectors to calculate the fundamental matrix
        vector<unsigned int>::const_iterator oit, cit;
        oit = i_old.begin();
        cit = i_cur.begin();

        for(; oit != i_old.end(); ++oit, ++cit)
        {
            const cv::KeyPoint &old_k = keyframeImgKeypoints[old_entry][*oit];
            const cv::KeyPoint &cur_k = keys[*cit];

            old_points.push_back(old_k.pt);
            cur_points.push_back(cur_k.pt);
        }

        cv::Mat oldMat(old_points.size(), 2, CV_32F, &old_points[0]);
        cv::Mat curMat(cur_points.size(), 2, CV_32F, &cur_points[0]);

        matchPoints = oldMat;
        framePoints = curMat;

        cv::Mat fundMat = fsolver.findFundamentalMat(oldMat, curMat, max_reprojection_error, min_Fpoints, NULL, true, ransac_probability, max_ransac_iterations);

        return fundMat;
    }

    return cv::Mat();
}

void FullSystem::addPointToClipperPath(int x, int y, ClipperLib::Path *path)
{
    ClipperLib::IntPoint ip;
    ip.X = x;
    ip.Y = y;
    path->push_back(ip);
}

bool FullSystem::notEnoughPointsCanBeProjected(const SE3 &transformation, float medianInvDepth)
{
    Mat33 K; K << Hcalib.fxl(), 0.0, Hcalib.cxl(), 0.0, Hcalib.fyl(), Hcalib.cyl(), 0.0, 0.0, 1.0;

    Mat33f RKi = (transformation.rotationMatrix() * K.inverse()).cast<float>();
    Vec3f t = transformation.translation().cast<float>();

    Vec3f pt1 = RKi * Vec3f(0, 0, 1) + t*medianInvDepth;
    float u1 = pt1[0] / pt1[2];
    float v1 = pt1[1] / pt1[2];
    int Ku1 = int(Hcalib.fxl() * u1 + Hcalib.cxl());
    int Kv1 = int(Hcalib.fyl() * v1 + Hcalib.cyl());

    Vec3f pt2 = RKi * Vec3f(wG[0], 0, 1) + t*medianInvDepth;
    float u2 = pt2[0] / pt2[2];
    float v2 = pt2[1] / pt2[2];
    int Ku2 = int(Hcalib.fxl() * u2 + Hcalib.cxl());
    int Kv2 = int(Hcalib.fyl() * v2 + Hcalib.cyl());

    Vec3f pt3 = RKi * Vec3f(wG[0], hG[0], 1) + t*medianInvDepth;
    float u3 = pt3[0] / pt3[2];
    float v3 = pt3[1] / pt3[2];
    int Ku3 = int(Hcalib.fxl() * u3 + Hcalib.cxl());
    int Kv3 = int(Hcalib.fyl() * v3 + Hcalib.cyl());

    Vec3f pt4 = RKi * Vec3f(0, hG[0], 1) + t*medianInvDepth;
    float u4 = pt4[0] / pt4[2];
    float v4 = pt4[1] / pt4[2];
    int Ku4 = int(Hcalib.fxl() * u4 + Hcalib.cxl());
    int Kv4 = int(Hcalib.fyl() * v4 + Hcalib.cyl());

    ClipperLib::Paths subj;
    ClipperLib::Paths clip;
    ClipperLib::Paths solution;

    ClipperLib::Path p;
    addPointToClipperPath(Ku1,Kv1, &p);
    addPointToClipperPath(Ku2,Kv2, &p);
    addPointToClipperPath(Ku3,Kv3, &p);
    addPointToClipperPath(Ku4,Kv4, &p);
    subj.push_back(p);

    ClipperLib::Path p2;
    addPointToClipperPath(0,0, &p2);
    addPointToClipperPath(wG[0],0, &p2);
    addPointToClipperPath(wG[0],hG[0], &p2);
    addPointToClipperPath(0,hG[0], &p2);
    clip.push_back(p2);

    ClipperLib::Clipper c;

    c.AddPaths(subj, ClipperLib::ptSubject, true);
    c.AddPaths(clip, ClipperLib::ptClip, true);
    c.Execute(ClipperLib::ctIntersection, solution, ClipperLib::pftNonZero, ClipperLib::pftNonZero);

    cout << "-----------------------" << endl;
    cout << wG[0] << " x " << hG[0] << endl;
    cout << Ku1 << " " << Kv1 << endl;
    cout << Ku2 << " " << Kv2 << endl;
    cout << Ku3 << " " << Kv3 << endl;
    cout << Ku4 << " " << Kv4 << endl;

    if (abs(Ku1) > 1e4 || abs(Ku2) > 1e4 || abs(Ku3) > 1e4 || abs(Ku4) > 1e4 || abs(Kv1) > 1e4 || abs(Kv2) > 1e4 || abs(Kv3) > 1e4 || abs(Kv4) > 1e4) return true;

    //assert((int)solution.size() == 1);
    if ((int)solution.size() < 1) return true;

    ClipperLib::Path p3 = solution.at(0);
    cout << abs(ClipperLib::Area(p3)) << endl;
    cout << "-----------------------" << endl;
    return (abs(ClipperLib::Area(p3)) <= 0.6 * hG[0] * wG[0]); // even 0.3 * hG[0] * wG[0] works for most sequences, but we wanted a more robust and efficient solution
}

void FullSystem::addGTSAMFactorsAndUpdate(FrameHessian* frame, bool betweenFactorsNeeded)
{
    chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();

    //gtsam::noiseModel::Diagonal::shared_ptr odometryNoise =
    //          gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << gtsam::Vector3::Constant(2),gtsam::Vector3::Constant(1)).finished());

    // GTSAM Between factor odometry noise - same in all directions
    gtsam::noiseModel::Isotropic::shared_ptr odometryBetweenFactorNoise = gtsam::noiseModel::Isotropic::Sigma(6,0.1);

    unsigned int numPhotometricFactors = 0;

    if (!betweenFactorsNeeded)
    {
        // add photometric factors for pairs of already marginalized keyframes which have been in the same local optimization window
        for (int lowFrameID: localOptimizationWindowKeyframes[frame->frameID])
        {
            // check if other frame is marginalized
            std::map <int, int>::iterator it = frameIDToMarginalizationIndex.find(lowFrameID);
            if (it != frameIDToMarginalizationIndex.end())
            {
                FrameHessian* targetFrame = marginalizedFrameHessians[it->second];

                SE3 reference = frame->shell->camToWorld;
                SE3 target = targetFrame->shell->camToWorld;
                // transformation from target to reference
                SE3 first = reference.inverse() * target;
                // transformation from reference to target
                SE3 second = target.inverse() * reference;

                // check if other frame has enough points
                if (targetFrame->pointHessiansMarginalized.size() > 300)
                {
                    //                    // DEBUG BEGIN
                    //                    double rotationAngle = 0.0;

                    //                    gtsam::KeySet ks = isam_.getFactorsUnsafe().keys();
                    //                    if (ks.find(lowFrameID) != ks.end())
                    //                    {
                    //                        gtsam::Values currentEstimate = isam_.calculateEstimate();
                    //                        SE3 optimizedTargetFramePose ((currentEstimate.at<gtsam::Pose3>(lowFrameID)).matrix());
                    //                        SE3 optimizedFirst = reference.inverse() * optimizedTargetFramePose;
                    //                        Eigen::AngleAxis<double> rotOptimizedFirst (first.unit_quaternion());
                    //                        rotationAngle = rotOptimizedFirst.angle();
                    //                        cout << "optimized rotation between frames " << frame->frameID << " and " << lowFrameID << ": " << rotationAngle << endl;
                    //                        cout << "optimized translation between frames " << frame->frameID << " and " << lowFrameID << ": " << optimizedFirst.translation().squaredNorm() << endl;
                    //                    }

                    //                Eigen::AngleAxis<double> rotFirst (first.unit_quaternion());
                    //                if (rotationAngle == 0.0) rotationAngle = rotFirst.angle();
                    //                cout << "rotation between frames " << frame->frameID << " and " << lowFrameID << ": " << rotFirst.angle() << endl;
                    //                cout << "translation between frames " << frame->frameID << " and " << lowFrameID << ": " << first.translation().squaredNorm() << endl;

                    //                //if (rotationAngle > M_PI/4) continue;
                    //                //if (rotationAngle > M_PI/3) continue; // usually works if coarsest level is 3
                    //                // DEBUG END

                    //            // between factors on demand
                    //            allBetweenFactors.add(gtsam::BetweenFactor<gtsam::Pose3>(frame->frameID, lowFrameID,
                    //                                                                     gtsam::Pose3(gtsam::Rot3(first.so3().unit_quaternion()), gtsam::Point3(ftr[0],ftr[1],ftr[2])),
                    //                                                                     odometryBetweenFactorNoise));

                    //            allBetweenFactors.add(gtsam::BetweenFactor<gtsam::Pose3>(lowFrameID, frame->frameID,
                    //                                                                     gtsam::Pose3(gtsam::Rot3(second.so3().unit_quaternion()), gtsam::Point3(str[0],str[1],str[2])),
                    //                                                                     odometryBetweenFactorNoise));

                    //            // add between factors

                    //            graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(frame->frameID, lowFrameID,
                    //                                                         gtsam::Pose3(gtsam::Rot3(first.so3().unit_quaternion()), gtsam::Point3(ftr[0],ftr[1],ftr[2])),
                    //                                                         odometryBetweenFactorNoise));

                    //            graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(lowFrameID, frame->frameID,
                    //                                                         gtsam::Pose3(gtsam::Rot3(second.so3().unit_quaternion()), gtsam::Point3(str[0],str[1],str[2])),
                    //                                                         odometryBetweenFactorNoise));

                    // assess if enough points project
                    if (notEnoughPointsCanBeProjected(first, medianInverseDepths[frameIDToMarginalizationIndex[lowFrameID]])) continue;
                    if (notEnoughPointsCanBeProjected(second, medianInverseDepths[frameIDToMarginalizationIndex[frame->frameID]])) continue;

                    // Photometric factors
                    // GTSAM photometric odometry noise - same for all residuals
                    // also try out lower noise - for example setting_outlierTH / 100
                    gtsam::noiseModel::Isotropic::shared_ptr odometryPhotometricFactorNoise =
                            gtsam::noiseModel::Isotropic::Sigma(frame->pointHessiansMarginalized.size() * patternNum, setting_outlierTH);
                    gtsam::noiseModel::Isotropic::shared_ptr odometryPhotometricFactorNoiseRev =
                            gtsam::noiseModel::Isotropic::Sigma(targetFrame->pointHessiansMarginalized.size() * patternNum, setting_outlierTH);

//                    // photometric factors on demand
//                    // coarsest level: pyrLevelsUsed - 1
//                    // make sure the factor graph at each level gets initialized in the FullSystem constructor
//                    for (int lvl = pyrLevelsUsed - (setting_numOfOmittedCoarsePyramidLevelsForLoopClosing + 1); lvl >= 0; lvl--)
//                    {
//                        allPhotometricFactors[lvl].add(gtsam::PhotometricFactor(lowFrameID, frame->frameID, odometryPhotometricFactorNoise,
//                                                                                frame, targetFrame, &Hcalib, lvl));
//                        allPhotometricFactors[lvl].add(gtsam::PhotometricFactor(frame->frameID, lowFrameID, odometryPhotometricFactorNoiseRev,
//                                                                                targetFrame, frame, &Hcalib, lvl));
//                    }

                    // add photometric factors
                    // possibly add factors at coarser pyramid levels as well
                    graph_.add(gtsam::PhotometricFactor(lowFrameID, frame->frameID, odometryPhotometricFactorNoise,
                                                        frame, targetFrame, &Hcalib, 0));
                    graph_.add(gtsam::PhotometricFactor(frame->frameID, lowFrameID, odometryPhotometricFactorNoiseRev,
                                                        targetFrame, frame, &Hcalib, 0));

                    // visualize photometric factors
                    for(IOWrap::Output3DWrapper* ow : outputWrapper)
                        ow->drawPhotometricFactor(frame, targetFrame);

                    cout << "photometric factors added for frames: " << frame->frameID << " and " << lowFrameID << endl;
                    numPhotometricFactors++;
                }
            }
        }
    }

    if (betweenFactorsNeeded || numPhotometricFactors < 3)
    {
        // add between factors for pairs of already marginalized keyframes which have been in the same local optimization window
        for (int lowFrameID: localOptimizationWindowKeyframes[frame->frameID])
        {
            // check if other frame is marginalized
            std::map <int, int>::iterator it = frameIDToMarginalizationIndex.find(lowFrameID);
            if (it != frameIDToMarginalizationIndex.end())
            {
                FrameHessian* targetFrame = marginalizedFrameHessians[it->second];

                SE3 reference = frame->shell->camToWorld;
                SE3 target = targetFrame->shell->camToWorld;
                // transformation from target to reference
                SE3 first = reference.inverse() * target;
                // transformation from reference to target
                SE3 second = target.inverse() * reference;

                // Between factors
                //                SE3 reference = frame->shell->camToWorld;
                //                SE3 target = targetFrame->shell->camToWorld;
                //                // transformation from target to reference
                //                SE3 first = reference.inverse() * target;
                //                // transformation from reference to target
                //                SE3 second = target.inverse() * reference;

                Vec3 ftr = first.translation();
                Vec3 str = second.translation();

                // add between factors

                graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(frame->frameID, lowFrameID,
                                                              gtsam::Pose3(gtsam::Rot3(first.so3().unit_quaternion()), gtsam::Point3(ftr[0],ftr[1],ftr[2])),
                        odometryBetweenFactorNoise));

                graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(lowFrameID, frame->frameID,
                                                              gtsam::Pose3(gtsam::Rot3(second.so3().unit_quaternion()), gtsam::Point3(str[0],str[1],str[2])),
                        odometryBetweenFactorNoise));

                // visualize between factors
                for(IOWrap::Output3DWrapper* ow : outputWrapper)
                    ow->drawBetweenFactor(frame, targetFrame);
            }
        }
    }

    // set prior factor for first frame
    if (frame->frameID == 0)
    {
        //gtsam::noiseModel::Diagonal::shared_ptr priorNoise =
        //        gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << gtsam::Vector3::Constant(1),gtsam::Vector3::Constant(0.1)).finished());

        // GTSAM prior noise - zero
        gtsam::noiseModel::Constrained::shared_ptr priorNoise = gtsam::noiseModel::Constrained::MixedSigmas(gtsam::Vector6::Constant(0));
        priorFactor_ = gtsam::PriorFactor<gtsam::Pose3>(frame->frameID, gtsam::Pose3(), priorNoise);

        graph_.add(priorFactor_);
        // coarsest level: pyrLevelsUsed - 1
//        for (int lvl = pyrLevelsUsed - (setting_numOfOmittedCoarsePyramidLevelsForLoopClosing + 1); lvl >= 0; lvl--)
//            allPhotometricFactors[lvl].add(priorFactor_);

        performISAMUpdate_ = true;
    }

    chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
    cout << "add GTSAM factors duration: " << chrono::duration_cast<chrono::microseconds>( t2 - t1 ).count() << endl;

    updateGTSAM();

    chrono::high_resolution_clock::time_point t3 = chrono::high_resolution_clock::now();
    cout << "global optimization after adding GTSAM factors duration: " << chrono::duration_cast<chrono::microseconds>( t3 - t2 ).count() << endl;
}

void FullSystem::visualizeGlobalOptimization()
{
    gtsam::Values currentEstimate = isam_.calculateEstimate();

    SE3 framePosesArray [marginalizedFrameHessians.size()];

    for (auto it = frameIDToMarginalizationIndex.begin(); it != frameIDToMarginalizationIndex.end(); ++it )
    {
        SE3 pose ((currentEstimate.at<gtsam::Pose3>(it->first)).matrix());
        framePosesArray[it->second] = pose;
    }

    std::vector<SE3> framePoses (framePosesArray, framePosesArray + sizeof framePosesArray / sizeof framePosesArray[0]);

    for(IOWrap::Output3DWrapper* ow : outputWrapper)
        ow->publishOptimizedKeyframes(marginalizedFrameHessians, framePoses, &Hcalib);

    //cout << "****************************************************" << endl;
    //currentEstimate.print("Current estimate: ");
}

void FullSystem::updateGTSAMtoCloseLoops()
{
//    // needed for the photometric optimization as a last step
//    size_t poseConstraintsBeginning = isam_.getFactorsUnsafe().size();

    chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();

    cout << "just before loop closing" << endl;

    time_t rawTime;
    time(&rawTime);
    string strTime (ctime(&rawTime));

    std::ofstream myfile;
    myfile.open ((strTime + "detected_loops.txt").c_str());
    myfile << std::setprecision(15);

    //for (gtsam::BetweenConstraint<gtsam::Pose3> bc : allBetweenConstraints) graph_.add(bc);
    //if (allBetweenConstraints.size() > 0) graph_.add(allBetweenConstraints[0]);

    if (allBetweenConstraints.size() < 10)
        for (gtsam::BetweenConstraint<gtsam::Pose3> bc : allBetweenConstraints)
        {
            graph_.add(bc);

            int key1 = (int) bc.key1();
            int key2 = (int) bc.key2();

            myfile << marginalizedFrameHessians[frameIDToMarginalizationIndex[key1]]->shell->timestamp << " "
                   << marginalizedFrameHessians[frameIDToMarginalizationIndex[key2]]->shell->timestamp << " added" << "\n";
        }

    else
    {
        std::vector<int> firstTimers, dejaVus;
        for (gtsam::BetweenConstraint<gtsam::Pose3> bc : allBetweenConstraints)
        {
            bool addBetweenConstraint = true;

            int key1 = (int) bc.key1();
            int key2 = (int) bc.key2();

            myfile << marginalizedFrameHessians[frameIDToMarginalizationIndex[key1]]->shell->timestamp << " "
                   << marginalizedFrameHessians[frameIDToMarginalizationIndex[key2]]->shell->timestamp;

            for (int i = 0; i < firstTimers.size(); i++)
                if ((abs(firstTimers[i]-key1) < 50) && (abs(dejaVus[i]-key2) < 50))
                {
                    addBetweenConstraint = false;
                    break;
                }

            if (addBetweenConstraint)
            {
                cout << "add a between constraint" << endl;
                graph_.add(bc);
                firstTimers.push_back(key1);
                dejaVus.push_back(key2);
                myfile << " added" << "\n";
            }
            else
            {
                cout << "don't add a between constraint" << endl;
                myfile << " not added" << "\n";
            }
        }
    }

    myfile.close();

    updateGTSAM();

    cout << "just after loop closing" << endl;

    chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
    cout << "global optimization with Between Constraints: " << chrono::duration_cast<chrono::microseconds>( t2 - t1 ).count() << endl;

//    // we have also tried out photometric optimization as a last step - it doesn't change a thing

//    size_t poseConstraintsEnd = isam_.getFactorsUnsafe().size();

//    gtsam::FactorIndices poseConstraintsToBeRemovedIndices;
//    for (size_t idx = poseConstraintsBeginning; idx < poseConstraintsEnd; idx++) poseConstraintsToBeRemovedIndices.push_back(idx);

//    for (gtsam::PhotometricFactor phf : allLoopPhotometricFactors) graph_.add(phf);

//    cin.get();
//    isam_.update(graph_, gtsam::Values(), poseConstraintsToBeRemovedIndices);
//    visualizeGlobalOptimization();
//    cout << "just after last photometric step" << endl;

}

//void FullSystem::updateGTSAMcoarseToFine()
//{
//    if (performISAMUpdate_)
//    {
//        // coarsest level: pyrLevelsUsed - 1
//        for (int lvl = pyrLevelsUsed - (setting_numOfOmittedCoarsePyramidLevelsForLoopClosing + 1); lvl >= 0; lvl--)
//        {
//            // construct vector of factor indices to be removed
//            gtsam::FactorIndices oldPhotometricFactorsToBeRemovedIndices;
////            size_t factorsEnd = isam_.getFactorsUnsafe().size();
////            for (size_t idx = 0; idx < factorsEnd; idx++) oldPhotometricFactorsToBeRemovedIndices.push_back(idx);
//            oldPhotometricFactorsToBeRemovedIndices.push_back((size_t) 1);

//            cout <<  "just before a multilevel update step at level " << lvl << endl;
////            // set evaluateNonlinearError true at ISAM2 initialization in the FullSystem constructor to obtain error before and after
////            gtsam::ISAM2Result result = isam_.update(allPhotometricFactors[lvl], gtsam::Values(), removeAllOldPhotometricFactorIndices);
////            std::cout << *result.errorBefore << " > " << *result.errorAfter << std::endl;
//            isam_.update(allPhotometricFactors[lvl], gtsam::Values(), oldPhotometricFactorsToBeRemovedIndices);
//            //isam_.update();

//            visualizeGlobalOptimization();

//            cout << "just after global photometric optimization at level " << lvl << endl;
//            //cin.get();
//        }
//    }
//    else cout << "no prior factor present" << endl;
//}

void FullSystem::updateGTSAM()
{
    // perfrom a single iSAM update step if the prior factor is already present
    if (performISAMUpdate_)
    {
        cout << "just before an update step" << endl;
//        gtsam::ISAM2Result result = isam_.update(graph_, initialEstimate_);
//        std::cout << *result.errorBefore << " > " << *result.errorAfter << std::endl;
        isam_.update(graph_, initialEstimate_);
        visualizeGlobalOptimization();

        // Clear the factor graph and the initial variable estimates for the next iteration
        graph_.resize(0);
        initialEstimate_.clear();
    }
    else cout << "no prior factor present" << endl;
}

// compute the difference between first and last frame
void FullSystem::firstLastFrameDiff()
{
    SE3 errorTransformation = frameHessians.back()->shell->camToWorld;

//    gtsam::Values currentEstimate = isam_.calculateEstimate();
//    SE3 firstPose ((currentEstimate.at<gtsam::Pose3>(0)).matrix());
//    SE3 lastPose ((currentEstimate.at<gtsam::Pose3>(marginalizedFrameHessians.size()-1)).matrix());
//    SE3 optimizedErrorTransformation = firstPose.inverse() * lastPose;

    SE3 optimizedErrorTransformation ((isam_.calculateEstimate().at<gtsam::Pose3>(marginalizedFrameHessians.size()-1)).matrix());

    //Eigen::AngleAxis<double> rotET (errorTransformation.unit_quaternion());
    //cout << "rotation error in radians: " << rotET.angle() << endl;

    cout << "keyframes first-last translation error squared norm: " << errorTransformation.translation().squaredNorm() << endl;
    cout << "optimized keyframes first-last translation error squared norm: " << optimizedErrorTransformation.translation().squaredNorm() << endl;
}

void FullSystem::closeLoops(FrameHessian* frame)
{
    chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();

    // extract the image corresponding to the frame being marginalized
    int nCols = wG[0];
    int nRows = hG[0];
    cv::Mat extractedImage = cv::Mat(nRows, nCols, CV_32FC3,frame->dI->data());
    cv::Mat frameImgMat = cv::Mat::zeros(nRows, nCols, CV_32FC1);
    cv::Mat channels [3];
    cv::split(extractedImage, channels);
    channels[0].convertTo(frameImgMat, CV_8U);
    //imshow("extracted", frameImgMat);

    // extract features from the frame image for matching against the database
    cv::Mat mask;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    (*ext)(frameImgMat,mask,keypoints,descriptors);

    // feature vector for the frame features
    ORBFeatures frameFeatures;

    for(int i = 0; i < descriptors.rows; i++)
    {
        frameFeatures.push_back(descriptors.row(i));
    }

    // query the database for matches to the currently marginalized keyframe among already marginalized keyframes
    if (db->size() >= 20)
    {
        DBoW2::QueryResults matches;
        db->query(frameFeatures, matches, 10);

        assert (localOptimizationWindowKeyframes.count(frame->frameID));
        std::vector<int> kfs = localOptimizationWindowKeyframes[frame->frameID];

        for (DBoW2::Result res : matches)
        {
            assert(marginalizedFrameHessians.size()>res.Id);
            assert(keyframeImgMats.size()>res.Id);
            assert(keyframeFeatures.size()>res.Id);
            assert(keyframeImgKeypoints.size()>res.Id);

            // loop detection

            // frame ID of match frame, also used as its GTSAM key
            int matchFrameID;

            if (res.Score > 0.01)
            {
                // look for the frame among the matginalized ones whose marginalization index equals the one of the match (res.Id)
                for (auto it = frameIDToMarginalizationIndex.begin(); it != frameIDToMarginalizationIndex.end(); it++)
                    if (it->second == res.Id)
                    {
                        matchFrameID = it->first;
                        break;
                    }

                // number of keyframes the marginalized keyframe and the database match are apart
                int iddiff = abs(matchFrameID - frame->frameID);

                // there are two reasons why a match is not considered relevant:
                // - the marginalized keyframe and the database match have been in the same local optimization window at any point in time
                // - the marginalized keyframe and the database match are too few keyframes apart
                //   (for a smaller id difference we impose a stricter condition on the database match score)
                if (
                        (std::find(kfs.begin(), kfs.end(), matchFrameID) == kfs.end())
                        && ((setting_closeShortLoops && res.Score > 0.015 && iddiff > 25) || iddiff > 100)
                        )
                {
                    FrameHessian* targetFrame = marginalizedFrameHessians[res.Id];
                    // reconstruct camera matrix
                    cv::Mat camMat = (cv::Mat_<double>(3,3) << Hcalib.fxl(), 0, Hcalib.cxl(), 0, Hcalib.fyl(), Hcalib.cyl(), 0, 0, 1);

                    // compute a fundamental matrix from the images of both keyframes, their keypoints and features
                    unsigned int di_levels = 0;
                    DBoW2::BowVector bv;
                    DBoW2::FeatureVector fv;
                    db->getVocabulary()->transform(frameFeatures, bv, fv, di_levels);

                    cv::Mat framePoints, matchPoints;

                    cv::Mat fundMat = computeFundamentalMatrix(res.Id, keypoints, frameFeatures, fv, framePoints, matchPoints);
                    //assert(!fundMat.empty());

                    if (fundMat.empty()) continue;

                    // compute the corresponding essential matrix
                    cv::Mat essMat = camMat.t() * fundMat * camMat;

                    // extract the rotation matrix from the essential one
                    Mat33 rot;
                    cv::Mat rotMat, transMat;
                    cv::recoverPose(essMat, framePoints, matchPoints, camMat, rotMat, transMat);

                    // convert to Eigen
                    cv::cv2eigen(rotMat,rot);

                    // compute the transformation matrix and energy from the new keyframe which is to be marginalized to a match
                    CoarseInitializer* keyfrCInit = new CoarseInitializer(nCols, nRows);
                    keyfrCInit->compOptSE3(frame, targetFrame, &Hcalib, rot);
                    SE3 trKM = keyfrCInit->thisToNext;
                    size_t teeKM = keyfrCInit->teeTotal;
                    delete keyfrCInit;

                    // compute a transformation matrix and energy in the opposite direction
                    CoarseInitializer* newKeyfrCInit = new CoarseInitializer(nCols, nRows);
                    newKeyfrCInit->compOptSE3(targetFrame, frame, &Hcalib, rot.transpose());
                    SE3 trMK = newKeyfrCInit->thisToNext;
                    size_t teeMK = newKeyfrCInit->teeTotal;
                    delete newKeyfrCInit;

                    Eigen::AngleAxis<double> rotKM (trKM.unit_quaternion());
                    double angleKM = rotKM.angle();
                    Eigen::AngleAxis<double> rotMK (trMK.unit_quaternion());
                    double angleMK = rotMK.angle();

                    SE3 errorTransformation = trMK * trKM;
                    Eigen::AngleAxis<double> rotET (errorTransformation.unit_quaternion());

                    // add factors for both ends of the loop
                    if ( ((setting_closeShortLoops && res.Score > 0.015 && iddiff > 25 && teeKM > 300 && teeMK > 300) || (iddiff > 100 && teeKM > 100 && teeMK > 100))
                         && angleKM < M_PI/2 && angleMK < M_PI/2
                         && errorTransformation.translation().squaredNorm() < 0.001
                         && rotET.angle() < 0.1
                         )
                    {
                        // DEBUG BEGIN
                        cout << "loop detected" << endl;
                        cout << iddiff << endl;
                        cout << res.Score << endl;
                        cout << teeKM << endl;
                        cout << teeMK << endl;
                        cout << angleKM << endl;
                        cout << angleMK << endl;
                        cout << "error transformation rotation angle and translation norm" << endl;
                        cout << rotET.angle() << endl;
                        cout << errorTransformation.translation().squaredNorm() << endl;

                        // visualize both ends of the loop in both directions
                        std::vector<FrameHessian*> frames;
                        std::vector<SE3> frameTransformations;
                        std::vector<IOWrap::Output3DWrapper::Color> frameColors;

                        // currently marginalized frame transformed towards match frame
                        frames.push_back(frame);
                        frameTransformations.push_back(trMK);
                        frameColors.push_back(IOWrap::Output3DWrapper::YELLOW);

                        // match frame
                        frames.push_back(targetFrame);
                        frameTransformations.push_back(SE3());
                        frameColors.push_back(IOWrap::Output3DWrapper::GREEN);

                        // match frame transformed towards currently marginalized frame
                        frames.push_back(targetFrame);
                        frameTransformations.push_back(trKM);
                        frameColors.push_back(IOWrap::Output3DWrapper::MAGENTA);

                        // currently marginalized frame
                        frames.push_back(frame);
                        frameTransformations.push_back(SE3());
                        frameColors.push_back(IOWrap::Output3DWrapper::CYAN);

                        for(IOWrap::Output3DWrapper* ow : outputWrapper)
                            ow->drawCamPoses(frames, frameTransformations, frameColors, &Hcalib);

                        //cin.get();

//                        // visualize projected points

//                        std::vector<cv::KeyPoint> referenceKeypoints, targetKeypoints, referenceKeypointsRev, targetKeypointsRev;
//                        std::vector<cv::DMatch> matches, matchesRev;

//                        Mat33f K = Mat33f::Zero();
//                        K(0,0) = Hcalib.fxl();
//                        K(1,1) = Hcalib.fyl();
//                        K(0,2) = Hcalib.cxl();
//                        K(1,2) = Hcalib.cyl();
//                        K(2,2) = 1;

//                        Mat33f KRKi = (K*trKM.rotationMatrix().cast<float>() * K.inverse()).cast<float>();
//                        Vec3f Kt = (K*trKM.translation().cast<float>()).cast<float>();

//                        Mat33f revKRKi = (K*trMK.rotationMatrix().cast<float>() * K.inverse()).cast<float>();
//                        Vec3f revKt = (K*trMK.translation().cast<float>()).cast<float>();

//                        cv::Mat img1; frameImgMat.copyTo(img1);
//                        cv::Mat img2; keyframeImgMats[res.Id].copyTo(img2);
//                        cv::Mat img3; keyframeImgMats[res.Id].copyTo(img3);
//                        cv::Mat img4; frameImgMat.copyTo(img4);
//                        int radius = 3;

//                        // draw projected points from cyan (currently marginalized frame) into green (match)
//                        std::vector<PointHessian*> vec = frame->pointHessiansMarginalized;
//                        for (int j=0;j<vec.size();j++) {
//                            float ou = vec[j]->u;
//                            float ov = vec[j]->v;
//                            float oid = vec[j]->idepth;

//                            Vec3f pt = KRKi * Vec3f(ou, ov, 1) + Kt*oid;
//                            float Ku = pt[0] / pt[2];
//                            float Kv = pt[1] / pt[2];
//                            float id = oid/pt[2];

//                            if (Ku > 1 && Kv > 1 && Ku < nCols-2 && Kv < nRows-2 && id > 0)
//                            {
//                                cv::circle(img1, cv::Point2f(ou, ov), radius, cv::Scalar::all(255));
//                                cv::circle(img2, cv::Point2f(Ku, Kv), radius, cv::Scalar::all(255));

//                                matches.push_back(cv::DMatch(referenceKeypoints.size(), referenceKeypoints.size(), 0));
//                                referenceKeypoints.push_back(cv::KeyPoint(cv::Point2f(ou, ov),1));
//                                targetKeypoints.push_back(cv::KeyPoint(cv::Point2f(Ku, Kv),1));
//                            }
//                        }

//                        // draw projected points from green into cyan
//                        std::vector<PointHessian*> otherVec = targetFrame->pointHessiansMarginalized;
//                        for (int j=0;j<otherVec.size();j++) {
//                            float ou = otherVec[j]->u;
//                            float ov = otherVec[j]->v;
//                            float oid = otherVec[j]->idepth;

//                            Vec3f pt = revKRKi * Vec3f(ou, ov, 1) + revKt*oid;
//                            float Ku = pt[0] / pt[2];
//                            float Kv = pt[1] / pt[2];
//                            float id = oid/pt[2];

//                            if(Ku > 1 && Kv > 1 && Ku < nCols-2 && Kv < nRows-2 && id > 0)
//                            {
//                                cv::circle(img3, cv::Point2f(ou, ov), radius, cv::Scalar::all(255));
//                                cv::circle(img4, cv::Point2f(Ku, Kv), radius, cv::Scalar::all(255));

//                                matchesRev.push_back(cv::DMatch(referenceKeypointsRev.size(), referenceKeypointsRev.size(), 0));
//                                referenceKeypointsRev.push_back(cv::KeyPoint(cv::Point2f(ou, ov),1));
//                                targetKeypointsRev.push_back(cv::KeyPoint(cv::Point2f(Ku, Kv),1));
//                            }
//                        }

//                        cv::Mat outputImgMat = cv::Mat::zeros(2*nRows, 2*nCols, CV_8UC1);
//                        img1.copyTo(outputImgMat.rowRange(0, nRows).colRange(0, nCols));
//                        img2.copyTo(outputImgMat.rowRange(0, nRows).colRange(nCols, 2*nCols));
//                        img3.copyTo(outputImgMat.rowRange(nRows, 2*nRows).colRange(0, nCols));
//                        img4.copyTo(outputImgMat.rowRange(nRows, 2*nRows).colRange(nCols, 2*nCols));
//                        cv::imshow( "Projected points", outputImgMat);

//                        time_t rawTime;
//                        time(&rawTime);
//                        string strTime (ctime(&rawTime));
//                        cv::imwrite("images/" + strTime + " " + to_string(frame->frameID) + " " + to_string(res.Id) + ".jpg", outputImgMat);
//                        //cv::waitKey(0);

//                        cv::Mat matchesImg, matchesImgRev;
//                        cv::drawMatches(frameImgMat, referenceKeypoints, keyframeImgMats[res.Id], targetKeypoints, matches, matchesImg,
//                                cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector< char >(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
//                        cv::drawMatches(keyframeImgMats[res.Id], referenceKeypointsRev, frameImgMat, targetKeypointsRev, matchesRev, matchesImgRev);

//                        cv::imshow( "Matches", matchesImg);
//                        cv::imshow( "Matches - reverse direction", matchesImgRev);
//                        cv::imwrite("images/" + strTime +
//                                    to_string(frame->frameID) + " " + to_string(res.Id) + " matches " + ".jpg", matchesImg);
//                        cv::imwrite("images/" + strTime +
//                                    to_string(frame->frameID) + " " + to_string(res.Id) + " reverse matches " + ".jpg", matchesImgRev);

//                        cv::waitKey(30);
//                        cin.get();
                        // DEBUG END

                        // Photometric factor noise
                        // also try out lower noise - for example setting_outlierTH / 100
                        gtsam::noiseModel::Isotropic::shared_ptr loopClosingPhotometricFactorNoise =
                                gtsam::noiseModel::Isotropic::Sigma(frame->pointHessiansMarginalized.size() * patternNum, setting_outlierTH);

                        gtsam::noiseModel::Isotropic::shared_ptr loopClosingPhotometricFactorNoiseRev =
                                gtsam::noiseModel::Isotropic::Sigma(targetFrame->pointHessiansMarginalized.size() * patternNum, setting_outlierTH);

//                        // photometric factors on demand
//                        // coarsest level: pyrLevelsUsed - 1
//                        for (int lvl = pyrLevelsUsed - (setting_numOfOmittedCoarsePyramidLevelsForLoopClosing + 1); lvl >= 0; lvl--)
//                        {
//                            allPhotometricFactors[lvl].add(gtsam::PhotometricFactor(matchFrameID, frame->frameID, loopClosingPhotometricFactorNoise,
//                                                                                    frame, targetFrame, &Hcalib, lvl, 100.0));
//                            allPhotometricFactors[lvl].add(gtsam::PhotometricFactor(frame->frameID, matchFrameID, loopClosingPhotometricFactorNoiseRev,
//                                                                                    targetFrame, frame, &Hcalib, lvl, 100.0));
//                        }


                        //gtsam::noiseModel::Diagonal::shared_ptr loopClosingNoise =
                        //         gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << gtsam::Vector3::Constant(0.2),gtsam::Vector3::Constant(0.1)).finished());

                        Vec3 transMK = trMK.translation();
                        Vec3 transKM = trKM.translation();

//                        // between constraints on demand
//                        allBetweenFactors.add(gtsam::BetweenConstraint<gtsam::Pose3>(gtsam::Pose3(gtsam::Rot3(trMK.so3().unit_quaternion()), gtsam::Point3(transMK[0],transMK[1],transMK[2])),
//                                                                         frame->frameID, matchFrameID));
//                        allBetweenFactors.add(gtsam::BetweenConstraint<gtsam::Pose3>(gtsam::Pose3(gtsam::Rot3(trKM.so3().unit_quaternion()), gtsam::Point3(transKM[0],transKM[1],transKM[2])),
//                                                                         matchFrameID, frame->frameID));

                        // add between constraints
//                        graph_.add(gtsam::BetweenConstraint<gtsam::Pose3>(gtsam::Pose3(gtsam::Rot3(trMK.so3().unit_quaternion()), gtsam::Point3(transMK[0],transMK[1],transMK[2])),
//                                                                         frame->frameID, matchFrameID));
//                        graph_.add(gtsam::BetweenConstraint<gtsam::Pose3>(gtsam::Pose3(gtsam::Rot3(trKM.so3().unit_quaternion()), gtsam::Point3(transKM[0],transKM[1],transKM[2])),
//                                                                         matchFrameID, frame->frameID));

//                        allBetweenConstraints.push_back(gtsam::BetweenConstraint<gtsam::Pose3>(gtsam::Pose3(gtsam::Rot3(trMK.so3().unit_quaternion()), gtsam::Point3(transMK[0],transMK[1],transMK[2])),
//                                                                         frame->frameID, matchFrameID));
                        allBetweenConstraints.push_back(gtsam::BetweenConstraint<gtsam::Pose3>(gtsam::Pose3(gtsam::Rot3(trKM.so3().unit_quaternion()), gtsam::Point3(transKM[0],transKM[1],transKM[2])),
                                                                         matchFrameID, frame->frameID));

                        // visualize between constraints
                        for(IOWrap::Output3DWrapper* ow : outputWrapper)
                            ow->drawBetweenConstraint(frame, targetFrame);

                        allLoopPhotometricFactors.push_back(gtsam::PhotometricFactor(matchFrameID, frame->frameID, loopClosingPhotometricFactorNoise,
                                                                                                              frame, targetFrame, &Hcalib, 0, 100));
                        allLoopPhotometricFactors.push_back(gtsam::PhotometricFactor(frame->frameID, matchFrameID, loopClosingPhotometricFactorNoiseRev,
                                                                                     targetFrame, frame, &Hcalib, 0, 100));
//                        // between factor noise
//                        // same noise in all directions, but a lot smaller than odometry noise
//                        gtsam::noiseModel::Isotropic::shared_ptr loopClosingBetweenFactorNoise = gtsam::noiseModel::Isotropic::Sigma(6,0.1);

//                        // add between factors
//                        graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(frame->frameID, matchFrameID,
//                                                                     gtsam::Pose3(gtsam::Rot3(trMK.so3().unit_quaternion()), gtsam::Point3(transMK[0],transMK[1],transMK[2])),
//                                                                     loopClosingBetweenFactorNoise));
//                        graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(matchFrameID, frame->frameID,
//                                                                     gtsam::Pose3(gtsam::Rot3(trKM.so3().unit_quaternion()), gtsam::Point3(transKM[0],transKM[1],transKM[2])),
//                                                                     loopClosingBetweenFactorNoise));

                        //updateGTSAM();

                        //updateGTSAMtoCloseLoop();
                    }
                }
            }
        }
    }

    // add keyframe to the database
    db->add(frameFeatures);

    // also its descriptors to the corresponding vector
    keyframeFeatures.push_back(frameFeatures);

    // as well as its rectified image matrix to the corresponding vector
    keyframeImgMats.push_back(frameImgMat);

    // and its keypoints to the corresponding vector
    keyframeImgKeypoints.push_back(keypoints);

    chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
    cout << "loop detection duration: " << chrono::duration_cast<chrono::microseconds>( t2 - t1 ).count() << endl;
}

void FullSystem::updateRelatedStructures(FrameHessian* frame)
{
    if (frame->pointHessiansMarginalized.size() > 300)
    {
        std::vector<float> marginalizedPointDepths;
        for (PointHessian* mp : frame->pointHessiansMarginalized) marginalizedPointDepths.push_back(mp->idepth_scaled);
        std::nth_element(marginalizedPointDepths.begin(), marginalizedPointDepths.begin() + marginalizedPointDepths.size()/2, marginalizedPointDepths.end());
        float medianInvDepth = marginalizedPointDepths[marginalizedPointDepths.size()/2];
        medianInverseDepths.push_back(medianInvDepth);
    }
    else medianInverseDepths.push_back(0);

    // add keyframe to marginalized keyframes structure
    marginalizedFrameHessians.push_back(frame);

    // add the (keyframe index, marginalization index) pair to the corresponding map
    frameIDToMarginalizationIndex.insert(std::make_pair(frame->frameID, marginalizedFrameHessians.size()-1));

    // provide initial pose estimate for global GTSAM optimization
    SE3 from = frame->shell->camToWorld;
    Vec3 fromtr = from.translation();
    initialEstimate_.insert(frame->frameID, gtsam::Pose3(gtsam::Rot3(from.so3().unit_quaternion()),gtsam::Point3(fromtr[0],fromtr[1],fromtr[2])));
}

} // namespace dso
