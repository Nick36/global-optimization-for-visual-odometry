All evaluation results correspond to the Monocular Visual Odometry Dataset developed at the Technical University of Munich (TUM MVO dataset), available at https://vision.in.tum.de/data/datasets/mono-dataset.

goDSO stands for `globally optimized DSO', i.e. the method developed in this Bachelor thesis.

ATE stands for Absolute Trajectory error, whereas RPE - for Relative Pose Error.

The files in the folders ./dso and ./godso represent trajectories recorded in test runs of the original and the globally optimized DSO algorithm. The files in the folder ./gt represent ground truth chunks of the trajectories recorded by means of a motion capture device. They are part of the supplementary material of DSO available at https://vision.in.tum.de/research/vslam/dso.

Notes on the format of the trajectory files:
Each line in the text file contains a single pose.
The format of each line is 'timestamp tx ty tz qx qy qz qw'.
timestamp (float) gives the number of seconds since the Unix epoch.
tx ty tz (3 floats) give the position of the optical center of the color camera with respect to the world origin as defined by the motion capture system.
qx qy qz qw (4 floats) give the orientation of the optical center of the color camera in form of a unit quaternion with respect to the world origin as defined by the motion capture system.

associate.py, evaluate_ate.py and evaluate_rpe.py are revised versions of the automated evaluation scripts available at https://vision.in.tum.de/data/datasets/rgbd-dataset/tools#evaluation.
compute_results.py employs functions defined in them in order to produces four files, each of which contains respectively the ATE of DSO (dso_ate.txt), the RPE of DSO (dso_rpe.txt), the ATE of goDSO (godso_ate.txt) and the RPE of goDSO (godso_rpe.txt).

Tthe application of a similar script produces the files in the folder ./ATE\ and\ RPE\ evaluation\ of\ the\ ORB-SLAM\ results\ recorded\ in\ the\ DSO\ supplementary\ material, each of which represents the ATE or RPE evaluation of a single trajectory from a file starting with the prefix "sequence" from the ./ORB_SLAM_FORWARD folder of the supplementary material of DSO available at https://vision.in.tum.de/research/vslam/dso. Each of these trajectories is the result of an application of ORB-SLAM with disabled explicit loop-closure detection and relocalization at a sequence of the TUM MVO dataset.

medians.py produces both files orb_ate.txt and orb_rpe.txt, the n-th line of which contains a single number representing the median of the evaluation values from the folder ./ATE\ and\ RPE\ evaluation\ of\ the\ ORB-SLAM\ results\ recorded\ in\ the\ DSO\ supplementary\ material for the n-th sequence of the TUM MVO dataset.

ate_cumulative_distribution_plot.py and rpe_cumulative_distribution_plot.py produce the cumulative distribution plots ATE\ cumulative\ distribution\ plot.png and RPE\ cumulative\ distribution\ plot.png, respectively, each of which juxtaposes the ATE or RPE of DSO, goDSO and ORB-SLAM. Both cumulative distribution plots with the suffix "DSO goDSO" provide a comparison solely between DSO and goDSO.
