## this is an example.

### VO tutorial 

[part1 PDF](pdf/VO-tutorial_up.pdf)

[part2 PDF](pdf/VO-tutorial_down.pdf) 

#### the pipeline

<img src="img/pipeline.png" alt="pipeline" style="zoom:60%;" />



##### bundle adjustment

[Introduction](https://blog.csdn.net/OptSolution/article/details/64442962)

[ (wiki)](https://en.wikipedia.org/wiki/Bundle_adjustment) 

 [paper- Bundle Adjustment — A Modern Synthesis](pdf/Bundle Adjustment — A Modern Synthesis.pdf)

##### pose = rotation + translation

##### RANSAC  

[tutorial with code](https://zhuanlan.zhihu.com/p/62238520)

##### Drift ([VO basic](https://zhuanlan.zhihu.com/p/23382110))

```
里程计一个很重要的特性，是它只关心局部时间上的运动，多数时候是指两个时刻间的运动。当我们以某种间隔对时间进行采样时，就可估计运动物体在各时间间隔之内的运动。由于这个估计受噪声影响，先前时刻的估计误差，会累加到后面时间的运动之上，这种现象称为漂移（Drift）。
```

<img src="img/dift.jpg" alt="dift" style="zoom:50%;" />



##### ICP 

##### photometric and depth error minimization.

##### Loop Closure

回环检测，简单的来说，就是来看看现在扫描的场景之前有没有遇到过。

<img src="img/loop_closure_show.png" alt="LOOP" style="zoom: 50%;" />

#before + after loop closure

- based on appearance (main stream) == measure the **similarity** of current and previous map:: 1-feature matching 2- bag of words 3-deep learning
- based on geometric info (accumulate error)

Accuracy + Recall to measure.





- Part I presented a historical review of the first 30 years of research in this
  field, a discussion on camera modeling and calibration, and a description of the main motion-estimation pipelines for both monocular and binocular schemes, outlining the pros and cons of each implementation

- Part II deals with feature matching, robustness, and applications. It reviews the main point-feature detectors used in VO and the different outlier-rejection schemes.



### VO CODE AND PACKAGES

https://github.com/klintan/vo-survey

https://github.com/tzutalin/awesome-visual-slam

https://github.com/zdzhaoyong/GSLAM



### DL based VO

```
Available pose estimation approaches are categorized into three groups: 

classical, 

hybrid, 

deep learning (DL)
```

![sota](img/sota.png)

> *from survey [A Survey on Deep Learning for Localization and Mapping](pdf/survey1.pdf)   Section3 Visual Odometry*



#### DL supervised VO

- ["Learning visual odometry with a convolutional network."](pdf/Learning Visual Odometry with a Convolutional Network.pdf) *VISAPP (1)*. 2015.    #first-work  [no code]
  -  firstly realises DL based VO through synchrony detection
     between image sequences and features. 

- Exploring Representation Learning With CNNs for Frame-to-Frame Ego-Motion Estimation,  IEEE Robotics and Automation Letters ( Volume: 1, Issue: 1, Jan. 2016)  [no code]

- &star; &star; &star; [DeepVO: Towards End-to-End Visual Odometry with Deep Recurrent Convolutional Neural Networks](pdf/deepVO.pdf)  [unofficial CODE](https://github.com/ChiWeiHsiao/DeepVO-pytorch) [project page](https://www.catalyzex.com/redirect?url=http://senwang.gitlab.io/DeepVO/)  ICRA2017   #monocular #RNN #CNN #end-to-end #supervised
  - directly estimating poses from raw monocular RGB images
  - use RNN+CNN

- “Learning monocular visual odometry through geometry aware curriculum learning,”  (ICRA), pp. 3549–3555, IEEE, 2019. [no code]
  - incorporates curriculum learning(train with increasing data complexity ) to improve the model generalization. 
  - propose geometric loss constraints
- “Distilling knowledge from a deep pose regressor network,”  (ICCV), pp. 263–272, 2019 [no code]
  - propose distillation to make model run real-tie on mobile devices.
- “Beyond tracking: Selecting memory and refining poses for deep visual
  odometry,” (CVPR), pp. 8575–8583, 2019.  [no code]
  - propose memory module to store global information
  - propose refining module for pose estimations, preserving context information.
- [Estimating Metric Scale Visual Odometry from Videos using 3D Convolutional Networks](pdf/3DCVO_IROS2019.pdf), IROS219 [code](https://github.com/alexanderkoumis/3dc_vo)

#### DL unsupervised VO

-  SfmLearner, [Unsupervised Learning of Depth and Ego-Motion from Video](pdf/sfm.pdf), CVPR2017 [good code](https://github.com/tinghuiz/SfMLearner?utm_source=catalyzex.com)
   -   jointly learns depth and camera ego-motion from video sequences, by utilizing view synthesis as a supervisory signal 
   -   there are basically** **two main problems that remained unsolved in the original work [29]:** 1) this monocular image based approach is **not able to provide pose estimates in a consistent global scale**. Due to the scale ambiguity, no physically meaningful global trajectory can be reconstructed, limiting its real use. 2) The photometric loss **assumes that the scene is static and without camera occlusions**.

#TO solve the scene dynamic. 

#use stereo images pairs to recover absolute scale

-  [UnDeepVO : Monocular Visual Odometry through Unsupervised Deep Learning](pdf/unDeepVO.pdf)  ICRA 2018 [no code]
-  “Unsupervised Learning of Monocular Depth Estimation and Visual Odometry with Deep Feature Reconstruction,” (CVPR), pp. 340–349, 2018 [good code](https://github.com/Huangying-Zhan/Depth-VO-Feat?utm_source=catalyzex.com)

training data uses stereo pairs, inference only uses monocular image. the training dataset (stereo) is different to the test set (mono).



#new geometric consistency loss,

- d, “Unsupervised scale-consistent depth and ego-motion learning from monocular video,” in NueIPS, pp. 35–45, 2019
  -  that enforces the consistency between predicted depth maps and reconstructed depth maps.
  -  the depth predictions are able to remain scale-consistent over consecutive frames,
     enabling pose estimates to be scale-consistent meanwhile



-  DeepMatchVO [Beyond Photometric Loss for Self-Supervised Ego-Motion Estimation  ](pdf/Beyond Photometric Loss for Self-Supervised Ego-Motion Estimation.pdf)    [code](https://github.com/hlzz/DeepMatchVO)    icra2019  #monocular #photometric+geometric #
   - this method uses CNN to predict the
     depth map of the target image and the relative motion of the
     target frame to other source frames. With depth and pose,
     the source image can be projected onto the target frame to
     synthesize the target view. It minimizes the error between
     the synthesis view and the actual image.
   - TWO INFO: photometric information
     like intensity and color from images [6], and geometric
     information computed from stable local keypoints [27].
   - explore intermediate geometric information such as
     pairwise matching and weak geometry generated automatically
     to improve the joint optimization for depth and motion.

-  Unsupervised Learning of Depth and Ego-Motion from Monocular Video Using 3D Geometric Constraints, CVPR2018  [code developed based on sfmlearner code](https://github.com/Shiaoming/vid2depth)



#environmental dynamics (e.g. pedestrians and vehicles) problem 

- GeoNet: Unsupervised Learning of Dense Depth, Optical Flow and Camera Pose (CVPR), 2018, pp. 1983-1992 [TF code](https://github.com/yzcjtr/GeoNet?utm_source=catalyzex.com)
  - divides its learning process into two sub-tasks by estimating **static** scene structures and motion **dynamics** separately

- “Learning monocular visual odometry with dense 3d mapping from dense 3d
  flow,” (IROS), pp. 6864–6871, IEEE, 2018.  [no code]
  - add a 2D flow + Depth map == 3D environment (point cloud map)  ->more accurate pose.



#GAN generates depth map

- “Ganvo: Unsupervised deep monocular visual odometry and depth estimation with generative adversarial networks,” ICRA2019
- “Sequential adversarial learning for self-supervised deep visual odometry,” (ICCV), 2019

#transforme

- [Transformer Guided Geometry Model for Flow-Based Unsupervised Visual Odometry](pdf/transformerVO.pdf)  Neural Computing and Applications (2020)  #unsupervised #transformer # [no code]
  - e a method consisting of two camera pose estimators that deal with the information from pairwise images and a short sequence of images respectively
  - a Transformer-like structure is adopted to build a geometry model over a local temporal window
  - ![](img/transformerVO.png)



### DL hybrid VO

#use depth estimation into conventional VO to recover the absolute scale metric of poses

- “Scale recovery for monocular visual odometry using depth estimated with deep convolutional neural fields,” ICCV, pp. 5870–5878, 2017. [no code]



- “Driven to distraction: Self-supervised distractor learning for robust monocular
  visual odometry in urban environments,” ICRA, 2018  [no code]
  - use depth maps and masks of moving objects into conventional VO
- DF VO “Visual odometry revisited: What should be learnt?,”   (ICRA), 2020.  [code only inference](https://github.com/Huangying-Zhan/DF-VO?utm_source=catalyzex.com)
  - use both depth and optical flow into conventional VO
- 



#combine  physical motion model with DNN

- “Backprop KF: Learning Discriminative Deterministic State Estimators,” (NeurIPS), 2016
- “Differentiable Particle Filters: End-to-End Learning with Algorithmic Priors,” in Robotics: Science and Systems, 2018.

- [Deep Virtual Stereo Odometry: Leveraging Deep Depth Prediction for Monocular Direct Sparse Odometry](pdf/DVSO.pdf) ECCV2018  

- D3VO: [Deep Depth, Deep Pose and Deep Uncertainty for Monocular Visual Odometry](pdf/D3VO.pdf)  CVPR2020    [[supp](https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Yang_D3VO_Deep_Depth_CVPR_2020_supplemental.pdf)] [[arXiv](http://arxiv.org/abs/2003.01060)] [[video](https://www.youtube.com/watch?v=bS9u28-2p7w)]  [no code]







### stereo VO

14 cvpr tutorial [stereo VO](pdf/VSLAM-Tutorial-CVPR14-A12-StereoVO.pdf)

 3D information reconstructed from the stereo image is used as the input of the translation estimation



- [Multi-Frame GAN: Image Enhancement for Stereo Visual Odometry in Low Light](pdf/multi-frameGAN.pdf) Proceedings of the Conference on Robot Learning, PMLR 100:651-660, 2020.  *#poor-light-condition*

  - takes two consecutive stereo image pairs and outputs the enhanced stereo images
    while preserving temporal and stereo consistency.
  - stereo based >> monocular based by by eliminating the scale ambiguity[5, 6, 7].
  - stereo, Both, feature-based methods [8, 1] and direct methods [4, 3] rely on image gradient-based key point extraction,
  - preserve spatial (i.e., inter-camera) and temporal consistency of the domain
    transfer
  - make use of unpaired datasets and explicitly address temporal and spatial coherence
    using optical flow

  

- [Stereo DSO: Large-Scale Direct Sparse Visual Odometry with Stereo Cameras](pdf/Wang_Stereo_DSO_Large-Scale_ICCV_2017_paper.pdf)

- [Event-based Stereo Visual Odometry](pdf/Event-based Stereo Visual Odometry.pdf)  arxiv 2020 [YouTube](https://youtu.be/3CPPs1gz04k), [Code](https://github.com/HKUST-Aerial-Robotics/ESVO.git)

  - use event camera data as input
  - In this paper we tackle the problem of stereo visual odometry
    (VO) with event cameras in natural scenes and arbitrary
    6-DoF motion. 

- [SOFT-SLAM: Computationally efficient stereo visual
  simultaneous localization and mapping for autonomous
  unmanned aerial vehicles∗](pdf/soft-slam.pdf)  *Journal of field robotics* 35.4 (2018): 578-595. *#stereo-vo* 

  - consists of  a novel stereo odometry algorithm relying on feature tracking (SOFT), which currently **ranks first among all stereo methods on the KITTI dataset**.

  - build a feature-based pose graph SLAM solution

  - ![soft](img/soft.png)

    a feature management==: feature extraction and matching, tracking, and selection.

    The entire ego-motion estimation is performed in two steps: first,
    the rotation is estimated using the five-point method, and second, the
    resulting rotation is used for estimating translation via minimization of
    reprojection errors in the image plane

- V-LOAM [Visual-lidar Odometry and Mapping: Low-drift, Robust, and Fast ](pdf/V-LOAM.pdf) ICRA2015  *#poor-light-condition*   [video demo]( https://www.youtube.com/watch?v=-6cwhPMAap8)

  - Visual odometry methods require moderate lighting conditions
    and fail if distinct visual features are insufficiently
  - motion estimation via moving lidars involves **motion distortion** in point clouds as range
    measurements are received at different times during continuous lidar motion.
  - TWO STAGEs: 1- Visual Odometry at a high frequency as the image frame rate (60Hz) to
    estimate motion. 2- lidar odometry at a low frequency (1 Hz) to refine motion estimates and remove distortion in the point clouds caused by drift of the VO.

- [ORB-SLAM2: an Open-Source SLAM System for Monocular, Stereo and RGB-D Cameras. ](pdf/orb-slam2.pdf) IEEE Transactions on Robotics 2017  [demo page](https://webdiis.unizar.es/~raulmur/orbslam/)  #stereo #monocular #mapping #loop-closing #localization #BA  

  - present ORB-SLAM2 a complete SLAM system
    for **monocular, stereo and RGB-D cameras,** including **map reuse,**
    **loop closing and relocalization capabilities.**
  - **Place recognition** is a key module
    of a SLAM system to close loops.

### monocular VO

```
monocular SLAM suffers from scale drift and may fail if
performing pure rotations in exploration.

depth is not observable from just one camera,
the scale of the map and estimated trajectory is unknown.

system bootstrapping require multi-view or
filtering techniques to produce an initial map as it cannot
be triangulated from the very first frame.


```



- [Lidar-Monocular Visual Odometry using Point and Line Features](pdf/Lidar-Monocular Visual Odometry using Point and Line Features.pdf)  ICRA2020

  - a novel lidar-monocular visual odometry approach using point and line features--previous point-based lidar visual method
  - During sensor fusion, we provide a robust method
    to extract the depth of the points and lines from the lidar data,
    and use the depth prior to guide camera tracking
  - 

- [CNN-SLAM: Real-time dense monocular SLAM with learned depth prediction  ](pdf/CNN-SLAM.pdf)  [code](https://www.catalyzex.com/redirect?url=https://github.com/zjiayao/cvpr17)  CVPR2017 

  - this paper investigates how **predicted depth maps** from a DNN can be deployed for accurate and dense **monocular reconstruction.**
  - use of depth prediction for estimating the absolute scale of
    the reconstruction, hence overcoming one of the major limitations of monocular SLAM.
  - camera pose estimation is inspired by the **key-frame approach** in [4] [LSD-SLAM: Large-Scale Direct Monocular SLAM.](pdf/LSD-SLAM.pdf) ECCV 2014.

- ORB-SLAM: A Versatile and Accurate Monocular SLAM System. IEEE Transactions on Robotics, vol. 31, no. 5, pp. 1147-1163, 2015

- **LSD-SLAM** [LSD-SLAM: Large-Scale Direct Monocular SLAM.](pdf/LSD-SLAM.pdf) ECCV 2014.

  - a direct (feature-less) monocular SLAM

- SVO ["SVO: Fast Semi-direct Monocular Visual Odometry,"]() ICRA, 2014.

- **DSO** [Direct Sparse Odometry](pdf/dso.pdf)  [page](https://vision.in.tum.de/research/vslam/dso)  [code](https://github.com/JakobEngel/dso_ros) TPAMI 2018 

  - Direct Sparse Odometry, J. Engel, V. Koltun, D. Cremers, In arXiv:1607.02565, 2016

    A Photometrically Calibrated Benchmark For Monocular Visual Odometry, J. Engel, V. Usenko, D. Cremers, In arXiv:1607.02555, 2016 

- D3VO: [Deep Depth, Deep Pose and Deep Uncertainty for Monocular Visual Odometry](pdf/D3VO.pdf)  CVPR2020    [[supp](https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Yang_D3VO_Deep_Depth_CVPR_2020_supplemental.pdf)] [[arXiv](http://arxiv.org/abs/2003.01060)] [[video](https://www.youtube.com/watch?v=bS9u28-2p7w)] 

- 

### RGB-D

- Real-Time Visual Odometry from Dense RGB-D Images, F. Steinbucker, J. Strum, D. Cremers, ICCV, 2011 [code](https://github.com/tzutalin/OpenCV-RgbdOdometry)

- [Dense Visual SLAM for RGB-D Cameras](https://github.com/tum-vision/dvo_slam)
  - [1]Dense Visual SLAM for RGB-D Cameras (C. Kerl, J. Sturm, D. Cremers), In Proc. of the Int. Conf. on Intelligent Robot Systems (IROS), 2013.
  - [2]Robust Odometry Estimation for RGB-D Cameras (C. Kerl, J. Sturm, D. Cremers), In Proc. of the IEEE Int. Conf. on Robotics and Automation (ICRA), 2013 
  - [3]Real-Time Visual Odometry from Dense RGB-D Images (F. Steinbruecker, J. Sturm, D. Cremers), In Workshop on Live Dense Reconstruction with Moving Cameras at the Intl. Conf. on Computer Vision (ICCV), 2011.

- [RTAB MAP - Real-Time Appearance-Based Mapping](https://github.com/introlab/rtabmap)
  - Online Global Loop Closure Detection for Large-Scale Multi-Session Graph-Based SLAM, 2014 
  - Appearance-Based Loop Closure Detection for Online Large-Scale and Long-Term Operation, 2013

- [Kintinuous](https://github.com/mp3guy/Kintinuous)
  - Real-time Large Scale Dense RGB-D SLAM with Volumetric Fusion, T. Whelan, M. Kaess, H. Johannsson, M.F. Fallon, J. J. Leonard and J.B. McDonald, IJRR '14

- [InfiniTAM∞ v2](http://www.robots.ox.ac.uk/~victor/infinitam/index.html)- 
  - . Very High Frame Rate Volumetric Integration of Depth Images on Mobile Device. IEEE Transactions on Visualization and Computer Graphics (Proceedings International Symposium on Mixed and Augmented Reality 2015

- [ElasticFusion](https://github.com/mp3guy/ElasticFusion)
  - ElasticFusion: Real-Time Dense SLAM and Light Source Estimation, T. Whelan, R. F. Salas-Moreno, B. Glocker, A. J. Davison and S. Leutenegger, IJRR '16 
  - ElasticFusion: Dense SLAM Without A Pose Graph, T. Whelan, S. Leutenegger, R. F. Salas-Moreno, B. Glocker and A. J. Davison, RSS '15
- [Co-Fusion](http://visual.cs.ucl.ac.uk/pubs/cofusion/index.html)
  - Co-Fusion: Real-time Segmentation, Tracking and Fusion of Multiple Objects. ICRA2017





### Global localization

Unlike VO, they do not suffer from a lack of initial poses and do not require access to camera parameters,good initialization, and handcrafted features

- PoseNet: [ A Convolutional Network for Real-Time 6-DOF Camera Relocalization](pdf/pose-net.pdf) [CODE](https://www.catalyzex.com/redirect?url=https://github.com/alexgkendall/caffe-posenet)  ICCV 2015  
  - present a robust and real-time monocular six degree
    of freedom relocalization system. Our system trains
    a convolutional neural network to regress the 6-DOF camera
    pose from a single RGB image in an end-to-end



- [ViPR](pdf/ViPR.pdf)   CVPR2020 workshop 
  - Absolute pose regression (APR) uses DL [63]
    as a cascade of convolution operators to learn poses only
    from 2D images. The pioneer PoseNet [33] has been extended
    by Bayesian approaches [31], long short-term memories
    (LSTMs) [77] and others [50, 26, 36, 11]. Recent APR
    methods such as VLocNet [72, 59] and DGRNets [42] introduce
    relative pose regression (RPR) to address the APR
  - This paper proposes a modular fusion technique for 6DoF pose estimation based on a PoseNetlike module and predictions of a relative module for VO.
  - learns both the absolute poses based on monocular (2D) imaging and the relative motion for the task of estimating VO.





### Dataset

Dataset for benchmark/test/experiment/evalutation

- [TUM Universtiy](http://vision.in.tum.de/data/datasets/rgbd-dataset/download)
- [KITTI Vision benchmark](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)
- [UNI-Freiburg](http://kaspar.informatik.uni-freiburg.de/~slamEvaluation/datasets.php)
- [ADVIO](https://github.com/AaltoVision/ADVIO)
- [Oxford RobotCar Dataset](https://robotcar-dataset.robots.ox.ac.uk/)
- [HRI (Honda Research Institute) Driving Datasets](https://usa.honda-ri.com/honda-driving-datasets)
- [Argoverse](https://www.argoverse.org/data.html)
- [nuScenes](https://www.nuscenes.org/)
- [Waymo Open Dataset](https://waymo.com/open/)
- [Lyft Level 5 AV Dataset 2019](https://level5.lyft.com/dataset/)
- [KAIST Urban Dataset](https://irap.kaist.ac.kr/dataset/)



### survey 

[A Survey on Deep Learning for Localization and Mapping](pdf/survey1.pdf)   Section3 Visual Odometry









