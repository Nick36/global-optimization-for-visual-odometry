/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


#pragma once
#include <pangolin/pangolin.h>
#include "boost/thread.hpp"
#include "util/MinimalImage.h"
#include "IOWrapper/Output3DWrapper.h"
#include <map>
#include <deque>


namespace dso
{

class FrameHessian;
class CalibHessian;
class FrameShell;


namespace IOWrap
{

class KeyFrameDisplay;

struct GraphConnection
{
	KeyFrameDisplay* from;
	KeyFrameDisplay* to;
	int fwdMarg, bwdMarg, fwdAct, bwdAct;
};


class PangolinDSOViewer : public Output3DWrapper
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    PangolinDSOViewer(int w, int h, bool startRunThread=true);
	virtual ~PangolinDSOViewer();

	void run();
	void close();

	void addImageToDisplay(std::string name, MinimalImageB3* image);
	void clearAllImagesToDisplay();


	// ==================== Output3DWrapper Functionality ======================
    virtual void publishGraph(const std::map<uint64_t,Eigen::Vector2i> &connectivity);
    virtual void publishKeyframes( std::vector<FrameHessian*> &frames, bool final, CalibHessian* HCalib);
    virtual void publishOptimizedKeyframes( std::vector<FrameHessian*> &frames, std::vector<SE3> &framePoses, CalibHessian* HCalib);
    virtual void publishCamPose(FrameShell* frame, CalibHessian* HCalib);
    virtual void drawCamPoses(std::vector<FrameHessian*> &frames, std::vector<SE3> &frameTransformations, std::vector<Color> &frameColors, CalibHessian* HCalib);
    virtual void clearCamPosesMap();
    virtual void drawPhotometricFactor(FrameHessian* from, FrameHessian* to);
    virtual void drawBetweenFactor(FrameHessian* from, FrameHessian* to);
    virtual void drawBetweenConstraint(FrameHessian* from, FrameHessian* to);


	virtual void pushLiveFrame(FrameHessian* image);
	virtual void pushDepthImage(MinimalImageB3* image);
    virtual bool needPushDepthImage();

	virtual void join();

    virtual void reset();
private:

	bool needReset;
	void reset_internal();
	void drawConstraints();

	boost::thread runThread;
	bool running;
	int w,h;



	// images rendering
	boost::mutex openImagesMutex;
	MinimalImageB3* internalVideoImg;
	MinimalImageB3* internalKFImg;
	MinimalImageB3* internalResImg;
	bool videoImgChanged, kfImgChanged, resImgChanged;



	// 3D model rendering
    boost::mutex model3DMutex;
	KeyFrameDisplay* currentCam;
	std::vector<KeyFrameDisplay*> keyframes;
    std::vector<Vec3f,Eigen::aligned_allocator<Vec3f>> allFramePoses;
	std::map<int, KeyFrameDisplay*> keyframesByKFID;
    std::vector<GraphConnection,Eigen::aligned_allocator<GraphConnection>> connections;

    // DSSLAM

    // map of <frameID, translation representing GTSAM optimized keyframe position> pairs used for optimized trajectory visualization
    std::map<int,Vec3f> frameIDtoOptimizedFramePose;

    // map of <keyframe shell id, camera pose optimized by GTSAM> pairs used for visualization
    std::map <int, SE3> frameShellIDtoOptimizedKeyframePose;

    // vector of keyframe display pointers representing GTSAM optimized keyframe poses
    std::vector<KeyFrameDisplay*> optimizedKeyframes;

    // vector of keyframe display pointers
    std::vector<KeyFrameDisplay*>  camPoses;
    // vector of their colors
    std::vector<Color>  camPosesColors;
    // vector of active point pairs, between which a line is to be drawn representing a PhotometricFactor
    std::vector<std::pair<Vec3f,Vec3f>,Eigen::aligned_allocator<std::pair<Vec3f,Vec3f>>> photometricFactorLines;

    // vector of all point pairs, between which a line is to be drawn representing a BetweenFactor
    std::vector<std::pair<Vec3f,Vec3f>,Eigen::aligned_allocator<std::pair<Vec3f,Vec3f>>> betweenFactorLines;

    // vector of all point pairs, between which a line is to be drawn representing a BetweenConstraint
    std::vector<std::pair<Vec3f,Vec3f>,Eigen::aligned_allocator<std::pair<Vec3f,Vec3f>>> betweenConstraintLines;



	// render settings
    bool settings_showKFCameras;
    bool settings_showOptimizedKFCameras;
	bool settings_showCurrentCamera;
	bool settings_showTrajectory;
    bool settings_showOptimizedTrajectory;
	bool settings_showActiveConstraints;
    bool settings_showAllGTSAMPhotometricFactors;
    bool settings_showAllGTSAMBeweenFactors;
    bool settings_showAllGTSAMBeweenConstraints;
	bool settings_showAllConstraints;
    bool settings_showPointCloud;
    bool settings_showOptimizedPointCloud;
    bool settings_showFirstLast;

	float settings_scaledVarTH;
	float settings_absVarTH;
	int settings_pointCloudMode;
	float settings_minRelBS;
	int settings_sparsity;


	// timings
	struct timeval last_track;
	struct timeval last_map;


	std::deque<float> lastNTrackingMs;
	std::deque<float> lastNMappingMs;
};



}



}
